/*
 * deepRL
 */

#include "deepQLearner.h"
#include <stdlib.h>
#include <time.h>

#define GAME_WIDTH 9
#define GAME_HEIGHT 6
#define GAME_SIZE GAME_WIDTH * GAME_HEIGHT
#define GAME_EPISODES 3000


enum catchAction
{
	ACTION_STAY  = 0,
	ACTION_LEFT  = 1,
	ACTION_RIGHT = 2,
	NUM_ACTIONS
};

static const char* actionStr( int action )
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

static int rand_x()
{
	return float(rand()) / float(RAND_MAX) * (GAME_WIDTH-1);
}

int main( int argc, char** argv )
{
	printf("deepRL-catch\n\n");
	srand(time(NULL));
	
	// create Q-Learner in Torch
	deepQLearner* dqn = deepQLearner::Create(GAME_SIZE, NUM_ACTIONS);
	
	if( !dqn )
	{
		printf("[deepRL]  failed to create dqn  %i  %i", GAME_SIZE, NUM_ACTIONS);
		return 0;
	}
	
	// allocate memory for the game input
	deepQLearner::Tensor* input_state = dqn->AllocTensor(GAME_SIZE);
	
	if( !input_state )
	{
		printf("[deepRL]  failed to allocate input tensor with %u elements", GAME_SIZE);
		return 0;
	}
	
	// game state
	int ball_x = rand_x();
	int ball_y = GAME_HEIGHT - 1;
	int play_x = (GAME_WIDTH / 2) + 1;
	
	
	// play a match of episodes
	int episodes_won = 0;
	int episode = 1;
	
	while(true)
	{
		// update the playing field
		for( int y=0; y < GAME_HEIGHT; y++ )
		{
			for( int x=0; x < GAME_WIDTH; x++ )
			{
				float cell_value = 0.0f;
				
				if( x == ball_x && y == ball_y )
					cell_value = 1.0f;
				else if( x == play_x && y == 0 )
					cell_value = 2.0f;
				
				input_state->cpuPtr[y*GAME_WIDTH+x] = cell_value;
			}
		}
		
		// ask the AI agent for their action
		int action = ACTION_STAY;
		
		if( !dqn->Forward(input_state, &action) )
		{
			printf("[deepRL]  dqn->Forward failed.\n");
			return 0;
		}
		
		printf("DQN action: %i %s\n", action, actionStr(action));
		
		// apply the agent's action, without going off-screen
		if( action == ACTION_LEFT && play_x > 0 )
			play_x--;
		else if( action == ACTION_RIGHT && play_x < (GAME_WIDTH-1) )
			play_x++;
		
		
		// make the ball fall
		ball_y--;
		
		
		// print screen
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


				
		// if the ball has reached the bottom, train & reset randomly
		if( ball_y <= 0 )
		{
			float reward = 0.0f;
			
			// if the agent caught the ball, give it a reward 
			if( ball_x == play_x ) 
			{
				reward = 1.0;
				episodes_won++;
				printf("WON episode %i\n", episode);
			}
			else
				printf("LOST episode %i\n", episode);
			
			printf("%i for %i  (%0.4f)\n", episodes_won, episode, float(episodes_won)/float(episode));
			episode++;
			
			// perform training
			if( !dqn->Backward(reward) )
			{
				printf("[deepRL]  dqn->Backward(%f) failed.\n", reward);
				return 0;
			}
			
			// reset the game for next episode
			ball_x = rand_x();
			ball_y = GAME_HEIGHT - 1;
			play_x = (GAME_WIDTH / 2) + 1;
		}
	}
	
	return 0;
}