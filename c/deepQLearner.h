/*
 * deepRL
 */

#ifndef __DEEP_Q_LEARNER_H_
#define __DEEP_Q_LEARNER_H_

#include <stdio.h>
#include <stdint.h>


struct lua_State;
struct THCState;
struct THFloatTensor;
struct THCudaTensor;


/*
 * deepQLearner
 */
class deepQLearner
{
public:
	/**
	 * Create a new instance of deepQLearner.
	 */
	static deepQLearner* Create( uint32_t numInputs, uint32_t numActions );
	
	/**
	 * Destructor
	 */
	~deepQLearner();

	/**
	 * Tensor wrapper for working with Torch/cuTorch.
	 */
	struct Tensor
	{
		THFloatTensor* cpuTensor;
		THCudaTensor*  gpuTensor;	// (THCudaTensor defined as THCudaFloatTensor in THCGenerateAllTypes)

		float* cpuPtr;
		float* gpuPtr;

		uint32_t width;
		uint32_t height;
		uint32_t depth;

		size_t elements;
		size_t size;
	};

	/**
	 * Allocate a Torch float tensor mapped to CPU/GPU.
	 */
	Tensor* AllocTensor( uint32_t elements /*uint32_t width, uint32_t height=1, uint32_t depth=1*/ );

	/**
	 * Run the next iteration of the network.
	 * From the input state, compute the action to follow.
	 */
	bool Forward( Tensor* input, int* action );

	/**
	 * Upon completion of an episode, apply a reward and learning.
	 */
	bool Backward( float reward );

private:
	deepQLearner();
	
	bool initLua();
	bool initNetwork( uint32_t numInputs, uint32_t numActions );
	
	lua_State* L;	/**< Lua/Torch7 operating environment */
	THCState*  THC;	/**< cutorch state */
	
	uint32_t mNumInputs;
	uint32_t mNumActions;
	
	Tensor* mRewardTensor;
	Tensor* mActionTensor;
};


#endif
