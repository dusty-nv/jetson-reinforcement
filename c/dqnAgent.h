/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __DEEP_Q_LEARNING_AGENT_H_
#define __DEEP_Q_LEARNING_AGENT_H_


#include "rlAgent.h"


/**
 * Deep Q-Learner Agent 
 */
class dqnAgent : public rlAgent
{
public:
	/**
	 * Create a new DQN agent training instance,
	 * the dimensions of a 2D image are expected.
	 */
	static dqnAgent* Create( uint32_t width, uint32_t height, 
					     uint32_t channels, uint32_t numActions );

	/**
	 * Destructor
	 */
	virtual ~dqnAgent();

	/**
	 * From the input state, predict the next action (inference)
	 * This function isn't used during training, for that see NextReward()
	 */
	virtual bool NextAction( Tensor* state, int* action );

	/**
	 * Next action with reward (training)
	 */
	virtual bool NextReward( float reward, bool end_episode );

	/**
	 * GetType
	 */
	virtual TypeID GetType() const 	{ return TYPE_DQN; }

	/**
 	 * TypeID
	 */
	const TypeID TYPE_DQN = TYPE_RL | (1 << 2);

protected:
	dqnAgent();
};


#endif
