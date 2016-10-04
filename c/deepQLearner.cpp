/*
 * deepRL
 */

#include "deepQLearner.h"
#include <string.h>

extern "C" 
{ 
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
#include <luaT.h>
}

#include <THC/THC.h>
#include "../cuda/cudaMappedMemory.h"

#define SCRIPT_FILENAME "test-DQN.lua"
#define SCRIPT_FUNC_NAME "update_network"



// constructor
deepQLearner::deepQLearner()
{
	L   = NULL;
	THC = NULL;
	
	mNumInputs  = 0;
	mNumActions = 0;
	
	mRewardTensor = NULL;
	mActionTensor = NULL;
}


// destructor
deepQLearner::~deepQLearner()
{
	if( L != NULL )
	{
		lua_close(L);
		L = NULL;
	}
}


// Create
deepQLearner* deepQLearner::Create( uint32_t numInputs, uint32_t numActions )
{
	if( numInputs == 0 || numActions == 0 )
		return NULL;
		
	// create new deepQLearner object
	deepQLearner* r = new deepQLearner();

	if( !r )
		return NULL;

	// initialize lua runtime and load DQN script
	if( !r->initLua() )
	{
		printf("[deepRL]  failed to initialize deepQLearner LUA runtime environment\n");
		delete r;
		return NULL;
	}
	
	if( !r->initNetwork(numInputs, numActions) )
	{
		printf("[deepRL]  failed to initialize deepQLearner network (inputs=%u, actions=%u)\n", numInputs, numActions);
		delete r;
		return NULL;
	}
		
	return r;
}


// new_Tensor
static deepQLearner::Tensor* new_Tensor()
{
	deepQLearner::Tensor* t = new deepQLearner::Tensor();

	t->cpuTensor = NULL;
	t->gpuTensor = NULL;
	t->cpuPtr    = NULL;
	t->gpuPtr    = NULL;
	t->size      = 0;

	return t;
}


// AllocTensor
deepQLearner::Tensor* deepQLearner::AllocTensor( uint32_t width/*, uint32_t height, uint32_t depth*/ )
{
	const uint32_t height = 1;
	const uint32_t depth  = 1;
	
	const size_t elem = width * height * depth;
	const size_t size = elem * sizeof(float);

	if( size == 0 )
		return NULL;

	// create Tensor wrapper object
	Tensor* t = new_Tensor();
	   

	// alloc CUDA mapped memory
	if( !cudaAllocMapped((void**)&t->cpuPtr, (void**)&t->gpuPtr, size) )
	{
		printf("[deepRL]  failed to alloc CUDA buffers for tensor size %zu bytes\n", size);
		return NULL;
	}


#if 0
	// set memory to default sequential pattern for debugging
	for( size_t n=0; n < elem; n++ )
		t->cpuPtr[n] = float(n);
#endif


	// alloc CPU tensor
	THFloatStorage* cpuStorage = THFloatStorage_newWithData(t->cpuPtr, elem);

	if( !cpuStorage )
	{
		printf("[deepRL]  failed to alloc CPU THFloatStorage\n");
		return NULL;
	}

	//long sizedata[2]   = { height, width };		// BUG:  should be reversed?
	//long stridedata[2] = { width, 1 };	// with YUV, { width, 3 }
       
	//THLongStorage* sizeStorage   = THLongStorage_newWithData(sizedata, 2);
	//THLongStorage* strideStorage = THLongStorage_newWithData(stridedata, 2);
    long sizedata[1]   = { width };
	long stridedata[1] = { 1 };
       
	THLongStorage* sizeStorage   = THLongStorage_newWithData(sizedata, 1);
	THLongStorage* strideStorage = THLongStorage_newWithData(stridedata, 1);
      
	  
	if( !sizeStorage || !strideStorage )
	{
		printf("[deepRL]  failed to alloc size/stride storage\n");
		return NULL;
	}

	t->cpuTensor = THFloatTensor_new();

	if( !t->cpuTensor )
	{
		printf("[deepRL]  failed to create CPU THFloatTensor()\n");
		return NULL;
	}

	THFloatTensor_setStorage(t->cpuTensor, cpuStorage, 0LL, sizeStorage, strideStorage);

	
	// alloc GPU tensor
	THCudaStorage* gpuStorage = THCudaStorage_newWithData(THC, t->gpuPtr, elem);

	if( !gpuStorage )
	{
		printf("[deepRL]  failed to alloc GPU THCudaStorage\n");
		return NULL;
	}

	t->gpuTensor = THCudaTensor_new(THC);

	if( !t->cpuTensor )
	{
		printf("[deepRL]  failed to create GPU THCudaTensor()\n");
		return NULL;
	}

	THCudaTensor_setStorage(THC, t->gpuTensor, gpuStorage, 0LL, sizeStorage, strideStorage);


	// save variables
	t->width    = width;
	t->height   = height;
	t->depth    = depth;
	t->elements = elem;
	t->size     = size;

	printf("[deepRL]  allocated %u x %u x %u float tensor (%zu bytes)\n", width, height, depth, size);
	return t;
}


// initLua
bool deepQLearner::initLua()
{
	// create LUA environment
	L = luaL_newstate();

	if( !L )
	{
		printf("[deepRL]  failed to create lua_State (luaL_newstate returned NULL)\n");
		return false;
	}

	printf("[deepRL]  created new lua_State\n");


	// load LUA libraries
	luaL_openlibs(L);
	printf("[deepRL]  opened LUA libraries\n");


	// load deepRL script
	printf("[deepRL]  loading '%s' \n", SCRIPT_FILENAME);
	const int res = luaL_dofile(L, SCRIPT_FILENAME);

	if( res == 1 ) 
	{
		printf("[deepRL]  error loading script: %s\n", SCRIPT_FILENAME);
		const char* luastr = lua_tostring(L,-1);

		if( luastr != NULL )
			printf("%s\n", luastr);
	}

	printf("[deepRL]  loading of '%s' complete.\n", SCRIPT_FILENAME);

	// get cuTorch state
	lua_getglobal(L, "cutorch");
	lua_getfield(L, -1, "_state");
	THC = (THCState*)lua_touserdata(L, -1);
	lua_pop(L, 2);

	if( !THC )
	{
		printf("[deepRL]  failed to retrieve cuTorch operating state\n");
		return false;
	}

	//printf("[deepRL]  cuTorch numDevices:  %i\n", THC->numDevices);
	return true;
}


// initNetwork
bool deepQLearner::initNetwork( uint32_t numInputs, uint32_t numActions )
{
	lua_getglobal(L, "init_network");

	lua_pushnumber(L, (double)numInputs);
	lua_pushnumber(L, (double)numActions);
	
	const int num_params = 2;
	const int num_result = 0;

	const int f_result = lua_pcall(L, num_params, num_result, 0);
	
	printf("[deepRL]  init_network() ran (result=%i)\n", f_result);

	if( f_result != 0 )
	{
		printf("[deepRL]  error running init_network()  %s\n", lua_tostring(L, -1));
		return false;
	}
	
	
	// create reward storage tensor
	mRewardTensor = AllocTensor(1);
	
	if( !mRewardTensor )
	{
		printf("[deepRL]  error allocating reward storage tensor, size=1\n");
		return false;
	}
	
	
	// create action storage tensor
	mActionTensor = AllocTensor(1);
	
	if( !mActionTensor )
	{
		printf("[deepRL]  error allocating action storage tensor, size=1\n");
		return false;
	}
	
	mNumInputs  = numInputs;
	mNumActions = numActions;
	
	return true;
}


// Forward
bool deepQLearner::Forward( Tensor* input, int* action )
{
	if( !input || !action )
		return false;
	
	#define SCRIPT_FORWARD "forward"
	
	lua_getglobal(L, SCRIPT_FORWARD);

	if( input != NULL )
		luaT_pushudata(L, (void*)input->cpuTensor, "torch.FloatTensor");
		//luaT_pushudata(L, (void*)input->gpuTensor, "torch.CudaTensor");
		
	//if( mActionTensor != NULL )
	//	luaT_pushudata(L, (void*)mActionTensor->cpuTensor, "torch.FloatTensor");

	const int num_params = 1;
	const int num_result = 1;

	const int f_result = lua_pcall(L, num_params, num_result, 0);
	//printf("[deepRL]  %s() ran (res=%i)\n", SCRIPT_FORWARD, f_result);

	if( f_result != 0 )
	{
		printf("[deepRL]  error running %s (result=%i)   %s\n", SCRIPT_FORWARD, f_result, lua_tostring(L, -1));
		return false;
	}

	// return value
	const double action_num = lua_tonumber(L, -1);
	lua_pop(L, 1);
	printf("action %f\n", float(action_num));
	
	//*action = (int)mActionTensor->cpuPtr[0] - 1;
	*action = (int)action_num - 1;
	return true;
}


// Backward
bool deepQLearner::Backward( float reward )
{
	#define SCRIPT_BACKWARD "backward"
	
	lua_getglobal(L, SCRIPT_BACKWARD);

	/*if( mRewardTensor != NULL )
	{
		mRewardTensor->cpuPtr[0] = reward;
		luaT_pushudata(L, (void*)mRewardTensor->cpuTensor, "torch.FloatTensor");
		//luaT_pushudata(L, (void*)input->gpuTensor, "torch.CudaTensor");
	}	*/
	lua_pushnumber(L, (double)reward);
	
	const int num_params = 1;
	const int num_result = 0;

	const int f_result = lua_pcall(L, num_params, num_result, 0);
	//printf("[deepRL]  %s() ran (res=%i)\n", SCRIPT_BACKWARD, f_result);

	if( f_result != 0 )
	{
		printf("[deepRL]  error running %s (result=%i)  %s\n", SCRIPT_BACKWARD, f_result, lua_tostring(L, -1));
		return false;
	}

	return true;
}
	
	
/*bool deepQLearner::updateNetwork( deepQLearner::Tensor* input, deepQLearner::Tensor* reward, deepQLearner::Tensor* output )
{
	lua_getglobal(L, SCRIPT_FUNC_NAME);

	if( input != NULL )
		luaT_pushudata(L, (void*)input->cpuTensor, "torch.FloatTensor");
		//luaT_pushudata(L, (void*)input->gpuTensor, "torch.CudaTensor");
		
	if( reward != NULL )
		luaT_pushudata(L, (void*)reward->cpuTensor, "torch.FloatTensor");

	if( output != NULL )
		luaT_pushudata(L, (void*)output->cpuTensor, "torch.FloatTensor");
	

	const int num_params = 3;
	const int num_result = 0;

	const int f_result = lua_pcall(L, num_params, num_result, 0);
	printf("[deepRL]  %s() ran (res=%i)\n", SCRIPT_FUNC_NAME, f_result);

	if( f_result != 0 )
	{
		printf("[deepRL]  error running %s   %s\n", SCRIPT_FUNC_NAME, lua_tostring(L, -1));
		return false;
	}

	return true;
}*/


