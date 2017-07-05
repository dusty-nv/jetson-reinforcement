/*
 * deepRL
 */

#ifdef USE_PYTHON

#include <stdio.h>

#include <Python.h>
#include <TH/TH.h>
#include <THC/THC.h>
#include <THP.h>
#include <THCP.h>

#include "cudaMappedMemory.h"


//extern "C" PyObject* THPFloatTensor_New(THFloatTensor *t);
//extern "C" PyObject* THCPFloatTensor_New(THCudaTensor *t);

extern THCState* state;


int main( int argc, char** argv )
{
	PyObject *pName, *pModule, *pDict, *pFunc;
	PyObject *pArgs, *pValue;

	printf("deepRL-console (python)\n");

	if (argc < 3) {
		fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
		return 1;
	}

	void* cpuPtr   = NULL;
	void* gpuPtr   = NULL;
	const int elem = 1024;
	const int size = sizeof(float) * elem;

	if( !cudaAllocMapped((void**)&cpuPtr, (void**)&gpuPtr, size) )
	{
		printf("[deepRL]  failed to alloc CUDA buffers for tensor size %zu bytes\n", size);
		return 0;
	}

	for( int i=0; i < elem; i++ )
		((float*)cpuPtr)[i] = float(i);

	printf("pyTorch THCState  0x%08X\n", state);

	Py_Initialize();

	// setup arguments to python script
	int py_argc = 3;
	char* py_argv[3];

	py_argv[0] = argv[1];
	py_argv[1] = "--foo";
	py_argv[2] = "--bar";

	PySys_SetArgv(py_argc, py_argv);
	//PySys_SetArgv(argc, argv);

	pName = PyString_FromString(argv[1]);

	// load the script
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	if( !pModule ) 
	{
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
		return 1;
	}

	// alloc CPU tensor storage descriptor
	THFloatStorage* cpuStorage = THFloatStorage_newWithData((float*)cpuPtr, elem);

	if( !cpuStorage )
	{
		printf("[deepRL]  failed to alloc CPU THFloatStorage\n");
		return 0;
	}

	// convey the size and dimensions of the tensor
	long sizedata[1]   = { elem };	// { height, width };
	long stridedata[1] = { 1 };		// { width, 3 }  (for YUV)
       
	THLongStorage* sizeStorage   = THLongStorage_newWithData(sizedata, 1/*2*/);
	THLongStorage* strideStorage = THLongStorage_newWithData(stridedata, 1/*2*/);

	// alloc CPU tensor
	THFloatTensor* cpuTensor = THFloatTensor_new();

	if( !cpuTensor )
	{
		printf("[deepRL]  failed to create CPU THFloatTensor()\n");
		return 0;
	}

	THFloatTensor_setStorage(cpuTensor, cpuStorage, 0LL, sizeStorage, strideStorage);

	// get the PyObject for CPU tensor
	PyObject* pyTensorCPU = THPFloatTensor_New(cpuTensor);

	if( !pyTensorCPU )
	{
		printf("[deepRL]  failed to get PyObject from THFloatTensor\n");
		return 0;
	}

	// confirm that the CUDA THCState global variable has been set
	// note this should have occurred after 'import torch' was run in python
	printf("pyTorch THCState  0x%08X\n", state);

	if( !state )
	{
		printf("[deepRL]  pyTorch THCState is NULL\n");
		return 0;
	}

	// alloc GPU tensor storage descriptor
	THCudaStorage* gpuStorage = THCudaStorage_newWithData(state, (float*)gpuPtr, elem);

	if( !gpuStorage )
	{
		printf("[deepRL]  failed to alloc GPU THCudaStorage\n");
		return 0;
	}

	// alloc GPU tensor and set the storage to our descriptor
	THCudaTensor* gpuTensor = THCudaTensor_new(state);

	if( !gpuTensor )
	{
		printf("[deepRL]  failed to create GPU THCudaTensor()\n");
		return 0;
	}

	THCudaTensor_setStorage(state, gpuTensor, gpuStorage, 0LL, sizeStorage, strideStorage);


	// get the pyObject for GPU tensor
	PyObject* pyTensorGPU = THCPFloatTensor_New(gpuTensor);

	if( !pyTensorGPU )
	{
		printf("[deepRL]  failed to get PyObject from THFloatTensor\n");
		return 0;
	}

	// call the function in the script
	pFunc = PyObject_GetAttrString(pModule, argv[2]);		/* pFunc is a new reference */
	const int numArgs = 1;	

	if (pFunc && PyCallable_Check(pFunc)) 
	{
		pArgs = PyTuple_New(numArgs);
		
		for (int i = 0; i < numArgs; ++i) 
		{
			/* pValue reference stolen here: */
			PyTuple_SetItem(pArgs, i, pyTensorGPU);
       	}

		// call the function
		pValue = PyObject_CallObject(pFunc, pArgs);

		Py_DECREF(pArgs);

		if (pValue != NULL) 
		{
			printf("Result of call: %ld\n", PyInt_AsLong(pValue));
			Py_DECREF(pValue);
		}
		else 
		{
			Py_DECREF(pFunc);
			Py_DECREF(pModule);
			PyErr_Print();
			fprintf(stderr,"Call failed\n");
			return 1;
		}
	}
	else 
	{
		if (PyErr_Occurred())
			PyErr_Print();

		fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
	}

	Py_XDECREF(pFunc);
	Py_DECREF(pModule);

	Py_Finalize();
	return 0;
}

#endif

