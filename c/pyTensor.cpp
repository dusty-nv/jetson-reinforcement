/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "pyTensor.h"
#include "pyTorch.h"

#include "cudaMappedMemory.h"


#if USE_PYTHON

//-------------------------------------------------------------------------------
extern THCState* state;
//-------------------------------------------------------------------------------


// constructor
static pyTensor* new_pyTensor()
{
	pyTensor* t = new pyTensor();

	t->cpuTensor = NULL;
	t->gpuTensor = NULL;

	t->pyTensorCPU = NULL;
	t->pyTensorGPU = NULL;

	t->cpuPtr = NULL;
	t->gpuPtr = NULL;

	t->width = 0;
	t->height = 0;
	t->depth = 0;

	t->elements = 0;
	t->size = 0;

	return t;
}


// Alloc
pyTensor* pyTensor::Alloc( uint32_t elements )
{
	if( elements == 0 )
		return NULL;

	return Alloc(elements, 1);
}


// Alloc
pyTensor* pyTensor::Alloc( uint32_t width, uint32_t height=1, uint32_t depth )
{
	if( width == 0 || height == 0 || depth == 0 )
		return NULL;


	// allocate new tensor wrapper object
	pyTensor* t = new_pyTensor();

	if( !t )
		return NULL;

	t->width    = width;
	t->height   = height;
	t->depth    = depth;
	t->elements = width * height * depth;
	t->size     = t->elements * sizeof(float);


	// allocate CUDA shared memory
	if( !cudaAllocMapped((void**)&t->cpuPtr, (void**)&t->gpuPtr, t->size) )
	{
		printf("[deepRL]  failed to alloc CUDA buffers for tensor size %zu bytes\n", t->size);
		return NULL;
	}

	// allocate CPU tensor storage descriptor
	THFloatStorage* cpuStorage = THFloatStorage_newWithData((float*)t->cpuPtr, t->elements);

	if( !cpuStorage )
	{
		printf("[deepRL]  failed to alloc CPU THFloatStorage\n");
		return 0;
	}

	// convey the size and dimensions of the tensor
	long sizedata[3]   = { t->depth, t->height, t->width };//{ t->elements };	// { height, width };
	long stridedata[3] = { t->width * t->height, t->width, 1 };//{ 1 };		// { width, 3 }  (for YUV)
       
	THLongStorage* sizeStorage   = THLongStorage_newWithData(sizedata, /*1*/3);
	THLongStorage* strideStorage = THLongStorage_newWithData(stridedata, /*1*/3);

	// alloc CPU tensor
	t->cpuTensor = THFloatTensor_new();

	if( !t->cpuTensor )
	{
		printf("[deepRL]  failed to create CPU THFloatTensor()\n");
		return NULL;
	}

	THFloatTensor_setStorage(t->cpuTensor, cpuStorage, 0LL, sizeStorage, strideStorage);

	// get the PyObject for CPU tensor
	t->pyTensorCPU = THPFloatTensor_New(t->cpuTensor);

	if( !t->pyTensorCPU )
	{
		printf("[deepRL]  failed to get PyObject from THFloatTensor\n");
		return NULL;
	}

	// confirm that the CUDA THCState global variable has been set
	// note this should have occurred after 'import torch' was run in python
	printf("[deepRL]  pyTorch THCState  0x%08X\n", state);

	if( !state )
	{
		printf("[deepRL]  pyTorch THCState is NULL\n");
		return NULL;
	}

	// alloc GPU tensor storage descriptor
	THCudaStorage* gpuStorage = THCudaStorage_newWithData(state, (float*)t->gpuPtr, t->elements);

	if( !gpuStorage )
	{
		printf("[deepRL]  failed to alloc GPU THCudaStorage\n");
		return NULL;
	}

	// alloc GPU tensor and set the storage to our descriptor
	t->gpuTensor = THCudaTensor_new(state);

	if( !t->gpuTensor )
	{
		printf("[deepRL]  failed to create GPU THCudaTensor()\n");
		return NULL;
	}

	THCudaTensor_setStorage(state, t->gpuTensor, gpuStorage, 0LL, sizeStorage, strideStorage);

	// get the pyObject for GPU tensor
	t->pyTensorGPU = THCPFloatTensor_New(t->gpuTensor);

	if( !t->pyTensorGPU )
	{
		printf("[deepRL]  failed to get PyObject from THFloatTensor\n");
		return NULL;
	}

	return t;
}

#endif

