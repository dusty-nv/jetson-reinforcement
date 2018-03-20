/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "rlAgent.h"
#include "pyTorch.h"
#include "pyTensor.h"


#ifdef USE_PYTHON

//-------------------------------------------------------------------------------
bool rlAgent::scriptingLoaded = false;
//-------------------------------------------------------------------------------


// constructor
rlAgent::rlAgent()
{	
	//mNewEpisode = true;
	mNumInputs  = 0;
	mNumActions = 0;
	mModuleName = "RL";
	mModuleObj  = NULL;
	
	mActionTensor = NULL;
	mRewardTensor = NULL;

	for( uint32_t n=0; n < NUM_FUNCTIONS; n++ )
	{
		mFunction[n] 	  = NULL;
		mFunctionArgs[n] = NULL;
	}
}


// destructor
rlAgent::~rlAgent()
{

}


// Create
rlAgent* rlAgent::Create( uint32_t numInputs, uint32_t numActions, const char* module, const char* nextAction, const char* nextReward, const char* loadModel, const char* saveModel )
{
	if( !module || numInputs == 0 || numActions == 0 || !module || !nextAction || !nextReward )
		return NULL;

	return Create(numInputs, 1, 1, numActions, module, nextAction, nextReward, loadModel, saveModel);
}


// Create
rlAgent* rlAgent::Create( uint32_t width, uint32_t height, uint32_t channels, uint32_t numActions, const char* module, const char* nextAction, const char* nextReward, const char* loadModel, const char* saveModel )
{
	if( !module || width == 0 || height == 0 || channels == 0 || numActions == 0 || !module || !nextAction || !nextReward )
		return NULL;

	// create new object
	rlAgent* rl = new rlAgent();

	if( !rl )
		return NULL;

	if( !rl->Init(width, height, channels, numActions, module, nextAction, nextReward, loadModel, saveModel) )
		return NULL;

	return rl;
}


// Init
bool rlAgent::Init( uint32_t width, uint32_t height, uint32_t channels, uint32_t numActions, const char* module, const char* nextAction, const char* nextReward, const char* loadModel, const char* saveModel, const char* optimizer, float learning_rate, uint32_t replay_mem, uint32_t batch_size, float gamma, float epsilon_start, float epsilon_end, float epsilon_decay, bool use_lstm, int lstm_size, bool allow_random, bool debug_mode)
{
	if( !module || width == 0 || height == 0 || channels == 0 || numActions == 0 || !module || !nextAction || !nextReward )
		return false;

	const uint32_t numInputs = width * height * channels;

	// format argument strings
	char inputsStr[32];
	char actionStr[32];
	char channelStr[32];
	char widthStr[32];
	char heightStr[32];
	char optimizerStr[32];
	char learning_rateStr[32];
	char replay_memStr[32];
	char batch_sizeStr[32]; 
	char gammaStr[32];
	char epsilon_startStr[32]; 
	char epsilon_endStr[32]; 
	char epsilon_decayStr[32]; 
	char use_lstmStr[32];
	char lstm_sizeStr[32];
	char allow_randomStr[32];
	char debug_modeStr[32]; 

	//sprintf(inputsStr, "--inputs=%u", numInputs);
	sprintf(widthStr, "--width=%u", width);
	sprintf(heightStr, "--height=%u", height);
	sprintf(channelStr, "--channels=%u", channels);
	sprintf(actionStr, "--actions=%u", numActions);
	sprintf(optimizerStr, "--optimizer=%s", optimizer);
	sprintf(learning_rateStr, "--learning_rate=%f", learning_rate);
	sprintf(replay_memStr, "--replay_mem=%u", replay_mem);
	sprintf(batch_sizeStr, "--batch_size=%u", batch_size);
	sprintf(gammaStr, "--gamma=%f", gamma);
	sprintf(epsilon_startStr, "--epsilon_start=%f", epsilon_start);
	sprintf(epsilon_endStr, "--epsilon_end=%f", epsilon_end);
	sprintf(epsilon_decayStr, "--epsilon_decay=%f", epsilon_decay);
	sprintf(use_lstmStr, "--use_lstm=%u", use_lstm);
	sprintf(lstm_sizeStr, "--lstm_size=%u", lstm_size);
	sprintf(allow_randomStr, "--allow_random=%u", allow_random);
	sprintf(debug_modeStr, "--debug_mode=%u", debug_mode);


	// set python command line
	int py_argc = 17;
	char* py_argv[17];

	py_argv[0] = (char*)module;
	py_argv[1] = actionStr;
	py_argv[2] = heightStr;
	py_argv[3] = widthStr;
	py_argv[4] = channelStr;
	py_argv[5] = optimizerStr;
	py_argv[6] = learning_rateStr;
	py_argv[7] = replay_memStr;
	py_argv[8] = batch_sizeStr;
	py_argv[9] = gammaStr;
	py_argv[10] = epsilon_startStr;
	py_argv[11] = epsilon_endStr;
	py_argv[12] = epsilon_decayStr;
	py_argv[13] = use_lstmStr;
	py_argv[14] = lstm_sizeStr;
	py_argv[15] = allow_randomStr;
	py_argv[16] = debug_modeStr;

	// load python module
	if( !LoadModule(module, py_argc, py_argv) )
		return false;

	// retrieve python RL functions
	PyObject* actionFunc = PyObject_GetAttrString((PyObject*)mModuleObj, nextAction);
	PyObject* rewardFunc = PyObject_GetAttrString((PyObject*)mModuleObj, nextReward);
	PyObject* loadFunc   = PyObject_GetAttrString((PyObject*)mModuleObj, loadModel);
	PyObject* saveFunc   = PyObject_GetAttrString((PyObject*)mModuleObj, saveModel);

	if( !actionFunc )
	{
		if (PyErr_Occurred())
			PyErr_Print();

		printf("[rlAgent]  failed to find function %s() in Python module '%s'\n", nextAction, module);
		return false;
	}

	if( !rewardFunc )
	{
		if (PyErr_Occurred())
			PyErr_Print();

		printf("[rlAgent]  failed to find function %s() in Python module '%s'\n", nextReward, module);
		return false;
	}

	// check that the retrieved functions are actually callable
	if( !PyCallable_Check(actionFunc) )
	{
		if (PyErr_Occurred())
			PyErr_Print();

		printf("[rlAgent]  %s() from Python module '%s' is not a callable function\n", nextAction, module);
		return false;
	}

	if( !PyCallable_Check(rewardFunc) )
	{
		if (PyErr_Occurred())
			PyErr_Print();

		printf("[rlAgent]  %s() from Python module '%s' is not a callable function\n", nextReward, module);
		return false;
	}

	if( loadFunc != NULL && !PyCallable_Check(loadFunc) )
	{
		if (PyErr_Occurred())
			PyErr_Print();

		printf("[rlAgent]  %s() from Python module '%s' is not a callable function\n", loadModel, module);
		loadFunc = NULL;
	}

	if( saveFunc != NULL && !PyCallable_Check(saveFunc) )
	{
		if (PyErr_Occurred())
			PyErr_Print();

		printf("[rlAgent]  %s() from Python module '%s' is not a callable function\n", saveModel, module);
		saveFunc = NULL;
	}

	mFunction[ACTION_FUNCTION] = (void*)actionFunc;
	mFunction[REWARD_FUNCTION] = (void*)rewardFunc;
	mFunction[LOAD_FUNCTION]   = (void*)loadFunc;
	mFunction[SAVE_FUNCTION]   = (void*)saveFunc;

	mFunctionName[ACTION_FUNCTION] = nextAction;
	mFunctionName[REWARD_FUNCTION] = nextReward;

	if( loadModel != NULL )
		mFunctionName[LOAD_FUNCTION] = loadModel;

	if( saveModel != NULL )
		mFunctionName[SAVE_FUNCTION] = saveModel;

	// allocate function arguments
	PyObject* actionArgs = PyTuple_New(1);
	PyObject* rewardArgs = PyTuple_New(2);

	if( !actionArgs || !rewardArgs )
	{
		printf("[rlAgent]  failed to allocated PyTuple for function arguments\n");
		return false;
	}

	mFunctionArgs[ACTION_FUNCTION] = (void*)actionArgs;
	mFunctionArgs[REWARD_FUNCTION] = (void*)rewardArgs;
	
	return true;
}


// LoadInterpreter
bool rlAgent::LoadInterpreter()
{
	if( scriptingLoaded )
		return true;

	Py_Initialize();
	
	scriptingLoaded = true;
	return true;
}


// LoadModule
bool rlAgent::LoadModule( const char* module )
{
	int argc = 0;
	return LoadModule(module, argc, NULL);
}


// LoadModule
bool rlAgent::LoadModule( const char* module, int argc, char** argv )
{
	if( !LoadInterpreter() )
		return false;

	if( argc > 0 ) 
		PySys_SetArgv(argc, argv);	

	PyObject* pyModuleName = PyString_FromString(module);

	if( !pyModuleName )
		return false;

	PyObject* pyModule = PyImport_Import(pyModuleName);
	Py_DECREF(pyModuleName);

	if( !pyModule )
	{
		PyErr_Print();
		printf("failed to load python module '%s'\n", module);
		return false;
	}

	mModuleName = module;
	mModuleObj  = (void*)pyModule;
	return true;
}


// LoadCheckpoint
bool rlAgent::LoadCheckpoint( const char* filename )
{
	if( !filename )
		return false;

	PyObject* pArgs = PyTuple_New(1);

	PyTuple_SetItem(pArgs, 0, PyString_FromString(filename));
	

	// call checkpoint function
	if( !mFunction[LOAD_FUNCTION] )
		return false;

	PyObject* pValue = PyObject_CallObject((PyObject*)mFunction[LOAD_FUNCTION], pArgs);

	Py_DECREF(pArgs);


	// check for success
	if( !pValue )
	{
		PyErr_Print();
		printf("[rlAgent]  call to %s() failed\n", mFunctionName[LOAD_FUNCTION].c_str());
		return false;
	}
	else
	{
		Py_DECREF(pValue);
		return true;
	}
}
	

// SaveCheckpoint
bool rlAgent::SaveCheckpoint( const char* filename )
{
	if( !filename )
		return false;

	PyObject* pArgs = PyTuple_New(1);

	PyTuple_SetItem(pArgs, 0, PyString_FromString(filename));
	

	// call checkpoint function
	if( !mFunction[SAVE_FUNCTION] )
		return false;

	PyObject* pValue = PyObject_CallObject((PyObject*)mFunction[SAVE_FUNCTION], pArgs);

	Py_DECREF(pArgs);


	// check for success
	if( !pValue )
	{
		PyErr_Print();
		printf("[rlAgent]  call to %s() failed\n", mFunctionName[SAVE_FUNCTION].c_str());
		return false;
	}
	else
	{
		Py_DECREF(pValue);
		return true;
	}
}


// NextAction
bool rlAgent::NextAction( Tensor* state, int* action )
{
	if( !state || !action )
		return false;

	// setup arguments to action function
	PyObject* pArgs = PyTuple_New(1); //(PyObject*)mFunctionArgs[ACTION_FUNCTION];

	PyTuple_SetItem(pArgs, 0, state->pyTensorGPU);
	
	// call action function
	PyObject* pValue = PyObject_CallObject((PyObject*)mFunction[ACTION_FUNCTION], pArgs);

	//Py_DECREF(pArgs);

	// check return value
	if( pValue != NULL )
	{
		//printf("[rlAgent]  result of %s(): %ld\n", mFunctionName[ACTION_FUNCTION].c_str(), PyInt_AsLong(pValue));
		*action = (int)PyInt_AsLong(pValue);		
		Py_DECREF(pValue);
	}
	else
	{
		PyErr_Print();
		printf("[rlAgent]  call to %s() failed\n", mFunctionName[ACTION_FUNCTION].c_str());
		return false;
	}

	return true;
}


// NextReward
bool rlAgent::NextReward( float reward, bool end_episode )
{
	// setup arguments to action function
	PyObject* pArgs = PyTuple_New(2); //(PyObject*)mFunctionArgs[REWARD_FUNCTION];

	//PyTuple_SetItem(pArgs, 0, state->pyTensorGPU);
	PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(reward));
	PyTuple_SetItem(pArgs, 1, PyBool_FromLong(/*mNewEpisode*/end_episode ? 1 : 0));

	//mNewEpisode = false;	// reset new_ep flag

	// call reward/training function
	PyObject* pValue = PyObject_CallObject((PyObject*)mFunction[REWARD_FUNCTION], pArgs);

	//Py_DECREF(pArgs);		// this invalidates the tensors

	// check return value
	/*if( pValue != NULL )
	{
		//printf("[rlAgent]  result of %s(): %ld\n", mRewardFunctionName.c_str(), PyInt_AsLong(pValue));
		*action = (int)PyInt_AsLong(pValue);		
		Py_DECREF(pValue);
	}
	else
	{
		PyErr_Print();
		printf("[rlAgent]  call to %s() failed\n", mFunctionName[REWARD_FUNCTION].c_str());
		return false;
	}*/

	if( !pValue )
	{
		PyErr_Print();
		printf("[rlAgent]  call to %s() failed\n", mFunctionName[REWARD_FUNCTION].c_str());
		return false;
	}

	return true;
}


// NextEpisode
/*void rlAgent::NextEpisode()
{
	mNewEpisode = true;
}*/

#endif

