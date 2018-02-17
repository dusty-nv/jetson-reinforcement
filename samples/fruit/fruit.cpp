/*
 * deepRL
 */

#include "deepRL.h"
#include "fruitEnv.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdlib.h>
#include <signal.h>


#define GAME_WIDTH 128
#define GAME_HEIGHT 128
#define NUM_CHANNELS 3

#define EPISODE_MAX_LENGTH 200



bool quit_signal = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		quit_signal = true;
	}
}


int main( int argc, char** argv )
{
	printf("deepRL-fruit\n\n");

	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	// create Fruit environment
	FruitEnv* fruit = FruitEnv::Create(GAME_WIDTH, GAME_HEIGHT, EPISODE_MAX_LENGTH);
	
	if( !fruit )
	{
		printf("[deepRL]  failed to create FruitEnv instance\n");
		return 0;
	}
	
	
	// create reinforcement learner agent in pyTorch
	dqnAgent* agent = dqnAgent::Create(GAME_WIDTH, GAME_HEIGHT, NUM_CHANNELS, NUM_ACTIONS);
	
	if( !agent )
	{
		printf("[deepRL]  failed to create deepRL instance  %ux%u  %u", GAME_WIDTH, GAME_HEIGHT, NUM_ACTIONS);
		return 0;
	}
	
	// allocate memory for the game input
	Tensor* input_state = Tensor::Alloc(GAME_WIDTH, GAME_HEIGHT, NUM_CHANNELS);
	
	if( !input_state )
	{
		printf("[deepRL]  failed to allocate input tensor with %ux%xu elements", GAME_WIDTH, GAME_HEIGHT);
		return 0;
	}
	
	
	// create openGL display
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( display != NULL )
	{
		texture = glTexture::Create(GAME_WIDTH, GAME_HEIGHT, GL_RGBA32F_ARB/*GL_RGBA8*/, NULL);

		if( !texture )
			printf("[deepRL]  failed to create openGL texture\n");
		
		display->SetTitle("Fruit RL");
	}
	else
		printf("[deepRL]  failed to create openGL display\n");
	
	
	// game loop
	while( !quit_signal )
	{
		// render fruit environment
		float* imgRGBA = fruit->Render();
		
		if( !imgRGBA )
		{
			printf("[deepRL]  failed to render FruitEnv\n");
			return 0;
		}
		
		// draw environment to display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				void* imgGL = texture->MapCUDA();

				if( imgGL != NULL )
				{
					cudaMemcpy(imgGL, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					CUDA(cudaDeviceSynchronize());
					texture->Unmap();
				}

				texture->Render(10,10);		
			}

			display->EndRender();
		}
		
		// ask the AI agent for their action
		/*int action = ACTION_STAY;
		
		if( !agent->NextAction(input_state, &action) )
		{
			printf("[deepRL]  agent->NextAction() failed.\n");
			return 0;
		}

		//printf("RL action: %i %s\n", action, actionStr(action));*/
		
		
		//if( !agent->NextReward(reward, end_episode) )
		//	printf("[deepRL]  agent->NextReward() failed\n");
	}
	
	return 0;
}
