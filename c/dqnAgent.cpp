/*
 * deepRL
 */

#include "dqnAgent.h"


// constructor
dqnAgent::dqnAgent()
{	

}


// destructor
dqnAgent::~dqnAgent()
{

}


// Create
dqnAgent* dqnAgent::Create( uint32_t width, uint32_t height, uint32_t channels, uint32_t numActions )
{
	if( width == 0 || height == 0 || channels == 0 || numActions == 0 )
		return NULL;
	
	dqnAgent* agent = new dqnAgent();

	if( !agent )
		return NULL;

	if( !agent->Init(width, height, channels, numActions, "DQN", DEFAULT_NEXT_ACTION, DEFAULT_NEXT_REWARD, DEFAULT_LOAD_MODEL, DEFAULT_SAVE_MODEL) )
		return NULL;

	return agent;
}



// NextAction
bool dqnAgent::NextAction( Tensor* state, int* action )
{
	if( !state || !action )
		return false;

	return rlAgent::NextAction(state, action);
}


// NextReward
bool dqnAgent::NextReward( float reward, bool end_episode )
{
	return rlAgent::NextReward(reward, end_episode);
}



