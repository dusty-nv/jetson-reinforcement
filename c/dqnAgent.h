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
	static dqnAgent* Create( uint32_t width, uint32_t height, uint32_t channels, uint32_t numActions, 
						const char* optimizer="RMSprop", float learning_rate=0.001, 
						uint32_t replay_mem=10000, uint32_t batch_size=64, float gamma=0.9, 
						float epsilon_start=0.9, float epsilon_end=0.05, float epsilon_decay=200,
			 			bool use_lstm=true, int lstm_size=256, bool allow_random=true, bool debug_mode=false);

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
