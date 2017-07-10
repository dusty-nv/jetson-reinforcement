/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __AI_AGENT_H_
#define __AI_AGENT_H_


#include "pyTensor.h"


/**
 * TypeID typedef
 */
typedef uint64_t TypeID;


/**
 * AI Agent base class, given a state predicts the best action.
 */
class aiAgent
{
public:
	/**
	 * Destructor
	 */
	virtual ~aiAgent();

	/**
	 * From the input state, predict the next action (inference)
	 * This function isn't used during training, for that see NextReward()
	 */
	virtual bool NextAction( Tensor* state, int* action ) = 0;

	/**
	 * Load model checkpoint
	 */
	virtual bool LoadCheckpoint( const char* filename ) = 0;

	/**
 	 * Save model checkpoint
	 */
	virtual bool SaveCheckpoint( const char* filename ) = 0;

	/**
	 * GetType
	 */
	virtual TypeID GetType() const 	{ return TYPE_AI; }

	/**
 	 * IsType
	 */
	bool IsType( TypeID type ) const	{ return ((GetType() & type) == type); }

	/**
 	 * TypeID
	 */
	const TypeID TYPE_AI = (1 << 0);

protected:
	aiAgent();
};


#endif
