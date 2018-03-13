/*
 * 1D deepRL example
 */

#include "deepRL.h"

#include <stdlib.h>
#include <time.h>

// Define DQN API settings
#define GAME_WIDTH   64
#define GAME_HEIGHT  64
#define NUM_CHANNELS 1
#define OPTIMIZER "RMSprop"
#define LEARNING_RATE 0.01f
#define REPLAY_MEMORY 10000
#define BATCH_SIZE 32
#define GAMMA 0.9f
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 200
#define ALLOW_RANDOM true
#define DEBUG_DQN false


// Turn visualization on or off
#define VISUALIZE false

// Set enviromoment variables
#define BALL_SIZE	8
#define BALL_SIZE2  (BALL_SIZE/2)
#define PLAY_SIZE   16
#define PLAY_SIZE2  (PLAY_SIZE/2)

// Set game history
#define GAME_HISTORY 20

bool gameHistory[GAME_HISTORY];
int  gameHistoryIdx = 0;

// Set actions
enum catchAction
{
	ACTION_STAY  = 0,
	ACTION_LEFT  = 1,
	ACTION_RIGHT = 2,
	NUM_ACTIONS
};


// Action choice output function
static const char* catchStr( int action )
{
	if( action == 0 )
	{
		return "STAY";
	}
	else if( action == 1 )
	{
		return "LEFT";
	}
	else if( action == 2 )
	{
		return "RIGHT";
	}
	
	return "NULL";
}

// Generate random x betwenn 0 and GAME_WIDTH-1
static inline int rand_x()
{
	return float(rand()) / float(RAND_MAX) * (GAME_WIDTH-1);
}


int main( int argc, char** argv )
{
	printf("deepRL-catch\n\n");
	srand(time(NULL));
	

	// Create reinforcement learner agent in pyTorch using API
	dqnAgent* agent = dqnAgent::Create(GAME_WIDTH, GAME_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, OPTIMIZER, LEARNING_RATE,
	REPLAY_MEMORY, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, ALLOW_RANDOM, DEBUG_DQN);
	
	// Check for agent creation 
	if( !agent )
	{
		printf("[deepRL]  failed to create deepRL instance  %ux%u  %u", GAME_WIDTH, GAME_HEIGHT, NUM_ACTIONS);
		return 0;
	}
	
	// Allocate memory for the game input
	Tensor* input_state = Tensor::Alloc(GAME_WIDTH, GAME_HEIGHT, NUM_CHANNELS);
	
	// Check for agent creation
	if( !input_state )
	{
		printf("[deepRL]  failed to allocate input tensor with %ux%xu elements", GAME_WIDTH, GAME_HEIGHT);
		return 0;
	}
	
	// Setup game state
	int ball_x = rand_x();
	int ball_y = GAME_HEIGHT - 1;
	int play_x = (GAME_WIDTH / 2) + 1;
	
	
	// Set initial state for accuracy
	int episodes_won = 0;
	int episode = 1;
	
	
	while(true)
	{
		// Update the playing field
		for( int y=0; y < GAME_HEIGHT; y++ )
		{
			for( int x=0; x < GAME_WIDTH; x++ )
			{
				float cell_value = 0.0f;
				
				if( (x >= ball_x - BALL_SIZE2) && (x <= ball_x + BALL_SIZE2) &&
				    (y >= ball_y - BALL_SIZE2) && (y <= ball_y + BALL_SIZE2) )
					cell_value = 1.0f;
				else if( (x >= play_x - PLAY_SIZE2) && (x <= play_x + PLAY_SIZE2) &&
					    (y == 0) )
					cell_value = 100.0f;
				
				for( int c=0; c < NUM_CHANNELS; c++ )
					input_state->cpuPtr[c*GAME_WIDTH*GAME_HEIGHT+y*GAME_WIDTH+x] = cell_value;
			}
		}
		
		// Ask the AI agent for their action
		int action = ACTION_STAY;
		
		// Get next action
		if( !agent->NextAction(input_state, &action) )
		{
			printf("[deepRL]  agent->NextAction() failed.\n");
			return 0;
		}

		//printf("RL action: %i %s\n", action, actionStr(action));
		
		const int prevDist = abs(play_x - ball_x);

		// Apply the agent's action, without going off-screen
		if( action == ACTION_LEFT && (play_x - PLAY_SIZE2) > 0 )
			play_x--;
		else if( action == ACTION_RIGHT && (play_x + PLAY_SIZE2) < (GAME_WIDTH-1) )
			play_x++;
		
		const int currDist = abs(play_x - ball_x);
		
		
		// Advance the simulation (make the ball fall)
		ball_y--;

#if VISUALIZE
		// print screen		
		printf("\n");

		for( int y=0; y < GAME_HEIGHT; y++ )
		{
			printf("|");
			
			for( int x=0; x < GAME_WIDTH; x++ )
			{
				if( x == ball_x && y == ball_y )
					printf("*");
				else if( x == play_x && y == 0 )
					printf("-");
				else
					printf(" ");
			}
			
			printf("|\n");
		}
#endif 
		
		// Compute reward
		float reward = 0.0f;

		if( currDist == 0 )
			reward = 1.0f;
		else if( currDist > prevDist )
			reward = -1.0f;
		else if( currDist < prevDist )
			reward = 1.0f;
		else if( currDist == prevDist )
			reward = 0.0f;


		// If the ball has reached the bottom, train & reset randomly
		bool end_episode = false;
		
		if( ball_y <= 0 )
		{
			bool ball_overlap = false;

			// Detect if the player paddle is overlapping with the ball
			for( int i=0; i < BALL_SIZE; i++ )
			{
				const int p = ball_x - BALL_SIZE2 + i;
				
				if( p >= play_x - PLAY_SIZE2 && p <= play_x + PLAY_SIZE2 )
				{
					ball_overlap = true;
					break;
				}
			}

			// If the agent caught the ball, give it a reward 
			if( ball_overlap ) 
			{
				reward = 1.0;
				episodes_won++;
				gameHistory[gameHistoryIdx] = true;
				printf("WON! episode %i\n", episode);
			}

			else
			{
				gameHistory[gameHistoryIdx] = false;
				printf("LOST episode %i\n", episode);
				reward = -1.0f;
			}

			// Print out statistics for tracking agent learning progress
			printf("%i for %i  (%0.4f)  ", episodes_won, episode, float(episodes_won)/float(episode));

			if( episode >= GAME_HISTORY )
			{
				uint32_t historyWins = 0;

				for( uint32_t n=0; n < GAME_HISTORY; n++ )
				{
					if( gameHistory[n] )
						historyWins++;
				}

				printf("%02u of last %u  (%0.4f)", historyWins, GAME_HISTORY, float(historyWins)/float(GAME_HISTORY));
			}

			printf("\n");
			gameHistoryIdx = (gameHistoryIdx + 1) % GAME_HISTORY;
			episode++;

			// Reset the game for next episode
			ball_x = rand_x();
			ball_y = GAME_HEIGHT - 1;
			play_x = (GAME_WIDTH / 2) + 1;
						
			// Flag as end of episode
			end_episode = true;
		}
		
		if( !agent->NextReward(reward, end_episode) )
			printf("[deepRL]  agent->NextReward() failed\n");
	}
	
	return 0;
}
