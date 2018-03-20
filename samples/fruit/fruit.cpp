/*
 * 2D deepRL example
 */

#include "deepRL.h"
#include "fruitEnv.h"
#include "commandLine.h"
#include "rand.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "cudaFont.h"
#include "cudaPlanar.h"

#include <stdlib.h>
#include <signal.h>


// Define DQN API Settings
#define DEFAULT_GAME_WIDTH   48
#define DEFAULT_GAME_HEIGHT  48
#define NUM_CHANNELS 3
#define OPTIMIZER "RMSprop"
#define LEARNING_RATE 0.01f
#define REPLAY_MEMORY 10000
#define BATCH_SIZE 32
#define GAMMA 0.9f
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 200
#define USE_LSTM true
#define LSTM_SIZE 256
#define ALLOW_RANDOM true
#define DEBUG_DQN false

// Environment variables
#define RENDER_ZOOM 4
#define DEFAULT_EPISODE_MAX_FRAMES 75
#define GAME_HISTORY 20

bool gameHistory[GAME_HISTORY];
int  gameHistoryIdx = 0;
int  gameHistoryMax = 0;

bool quit_signal = false;

// Function to catch interupt and quit program
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

	// Catch quit signal to stop game
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	// Parse command line
	commandLine cmdLine(argc, argv);

	const int gameWidth   = cmdLine.GetInt("width",  DEFAULT_GAME_WIDTH);
	const int gameHeight  = cmdLine.GetInt("height", DEFAULT_GAME_HEIGHT);
	const int epMaxFrames = cmdLine.GetInt("episode_max_frames", DEFAULT_EPISODE_MAX_FRAMES);


	// Create Fruit environment
	FruitEnv* fruit = FruitEnv::Create(gameWidth, gameHeight, epMaxFrames);
	
	if( !fruit )
	{
		printf("[deepRL]  failed to create FruitEnv instance\n");
		return 0;
	}
	
	
	// Create reinforcement learner agent in pyTorch
	dqnAgent* agent = dqnAgent::Create(gameWidth, gameHeight, NUM_CHANNELS, NUM_ACTIONS, 
								OPTIMIZER, LEARNING_RATE, REPLAY_MEMORY, BATCH_SIZE, 
								GAMMA, EPS_START, EPS_END, EPS_DECAY, 
								USE_LSTM, LSTM_SIZE, ALLOW_RANDOM, DEBUG_DQN);
	
	if( !agent )
	{
		printf("[deepRL]  failed to create deepRL instance  %ux%u  %u", gameWidth, gameHeight, NUM_ACTIONS);
		return 0;
	}
	

	// Allocate memory for the game input
	Tensor* input_tensor = Tensor::Alloc(gameWidth, gameHeight, NUM_CHANNELS);
	
	// Check for proper
	if( !input_tensor )
	{
		printf("[deepRL]  failed to allocate input tensor with %ux%xu elements", gameWidth, gameHeight);
		return 0;
	}
	
	
	// Create OpenGL display
	glDisplay* display = glDisplay::Create("Fruit DQN", 0.1f, 0.1f, 0.1f);
	glTexture* texture = NULL;
	
	// Continue Display Initialization
	if( display != NULL )
	{
		texture = glTexture::Create(gameWidth, gameHeight, GL_RGBA32F_ARB/*GL_RGBA8*/, NULL);

		if( !texture )
			printf("[deepRL]  failed to create openGL texture\n");
	}
	else
		printf("[deepRL]  failed to create openGL display\n");
	
	
	// Create font object
	cudaFont* font = cudaFont::Create();
	
	// Check for font object creation
	if( !font )
		printf("failed to create cudaFont object\n");


	// Set global variables for tracking agent progress
	uint32_t episode_count = 0;
	uint32_t episode_wins  = 0;

	float reward = 0.0f;


	// Run game loop
	while( !quit_signal )
	{
		// Render fruit environment
		float* imgRGBA = fruit->Render();
		
		// Check for proper enviroment configuration 
		if( !imgRGBA )
		{
			printf("[deepRL]  failed to render FruitEnv\n");
			return 0;
		}
		
		if( font != NULL )
		{
			/*char str[256];
			sprintf(str, "%f", reward);

			font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, gameWidth, gameHeight,
							    str, 0, 0, make_float4(0.0f, 0.75f, 1.0f, 255.0f));*/
		}

		// Draw environment to display
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

				texture->Render(50, 50, gameWidth * RENDER_ZOOM, gameHeight * RENDER_ZOOM);		
			}

			display->EndRender();
		}
		
		// Convert from RGBA to pyTorch tensor format (CHW)
		CUDA(cudaRGBAToPlanarBGR((float4*)imgRGBA, gameWidth, gameHeight,
							(float*)input_tensor->gpuPtr, gameWidth, gameHeight));
	
	
		// Ask the agent for their action
		int action = 0;

		if( !agent->NextAction(input_tensor, &action) )
			printf("[deepRL]  agent->NextAction() failed.\n");

		if( action < 0 || action >= NUM_ACTIONS )
			action = ACTION_NONE;

		// Provide the agent's action to the environment
		const bool end_episode = fruit->Action((AgentAction)action, &reward);
		
		// End episode and log the outcome
		if( end_episode )
		{
			if( reward >= fruit->GetMaxReward() )
			{
				gameHistory[gameHistoryIdx] = true;
				episode_wins++;
			}
			else
				gameHistory[gameHistoryIdx] = false;

			gameHistoryIdx = (gameHistoryIdx + 1) % GAME_HISTORY;
			episode_count++;
		}

		printf("action = %s  reward = %+0.4f %s wins = %03u of %03u (%0.2f)   ", 
			  FruitEnv::ActionToStr((AgentAction)action), 
			  reward, end_episode ? "EOE" : "   ",  
			  episode_wins, episode_count, float(episode_wins)/float(episode_count));

		if( episode_count >= GAME_HISTORY )
		{
			uint32_t historyWins = 0;

			for( uint32_t n=0; n < GAME_HISTORY; n++ )
			{
				if( gameHistory[n] )
					historyWins++;
			}

			if( historyWins > gameHistoryMax )
				gameHistoryMax = historyWins;

			printf("%02u of last %u  (%0.2f)  (max=%0.2f)", historyWins, GAME_HISTORY, float(historyWins)/float(GAME_HISTORY), float(gameHistoryMax)/float(GAME_HISTORY));
		}

		printf("\n");

		// Train the agent with the reward
		if( !agent->NextReward(reward, end_episode) )
			printf("[deepRL]  agent->NextReward() failed\n");
	}
	
	return 0;
}
