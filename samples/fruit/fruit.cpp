/*
 * deepRL
 */

#include "deepRL.h"
#include "fruitEnv.h"
#include "rand.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "cudaFont.h"
#include "cudaPlanar.h"

#include <stdlib.h>
#include <signal.h>


#define GAME_WIDTH 128
#define GAME_HEIGHT 128
#define RENDER_ZOOM 4
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
	Tensor* input_tensor = Tensor::Alloc(GAME_WIDTH, GAME_HEIGHT, NUM_CHANNELS);
	
	if( !input_tensor )
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
	
	
	// create font object
	cudaFont* font = cudaFont::Create();
	
	if( !font )
		printf("failed to create cudaFont object\n");


	// global variables for tracking agent progress
	uint32_t episode_count = 0;
	uint32_t episode_wins  = 0;

	float reward = 0.0f;


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
		
		if( font != NULL )
		{
			/*char str[256];
			sprintf(str, "%f", reward);

			font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, GAME_WIDTH, GAME_HEIGHT,
							    str, 0, 0, make_float4(0.0f, 0.75f, 1.0f, 255.0f));*/
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

				texture->Render(50, 50, GAME_WIDTH * RENDER_ZOOM, GAME_HEIGHT * RENDER_ZOOM);		
			}

			display->EndRender();
		}
		
		// convert from RGBA to pyTorch tensor format (CHW)
		CUDA(cudaRGBAToPlanarBGR((float4*)imgRGBA, GAME_WIDTH, GAME_HEIGHT,
							(float*)input_tensor->gpuPtr, GAME_WIDTH, GAME_HEIGHT));
	
	
		// ask the agent for their action
		int action = ACTION_NONE;	//rand(0, NUM_ACTIONS);

		if( !agent->NextAction(input_tensor, &action) )
			printf("[deepRL]  agent->NextAction() failed.\n");

	
		// provide the agent's action to the environment
		const bool end_episode = fruit->Action((AgentAction)action, &reward);
		
		if( end_episode )
		{
			if( reward >= fruit->GetMaxReward() )
				episode_wins++;

			episode_count++;
		}

		printf("action = %s  reward = %f  %s  wins = %u of %u (%f)\n", 
			  FruitEnv::ActionToStr((AgentAction)action), 
			  reward, end_episode ? "EOE" : "",
			  episode_wins, episode_count, float(episode_wins)/float(episode_count));


		// train the agent with the reward
		if( !agent->NextReward(reward, end_episode) )
			printf("[deepRL]  agent->NextReward() failed\n");
	}
	
	return 0;
}
