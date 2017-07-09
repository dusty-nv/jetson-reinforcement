/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __PY_TENSOR_H_
#define __PY_TENSOR_H_

#include <stdio.h>
#include <stdint.h>

#ifdef USE_PYTHON

struct THFloatTensor;
struct THCudaTensor;

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif


/**
 * Tensor wrapper for working with Torch/pyTorch.
 */
struct pyTensor
{
	/**
	 * Allocate a Torch float tensor mapped to CPU/GPU.
	 */
	static pyTensor* Alloc( uint32_t elements );

	/**
	 * Allocate a Torch float tensor mapped to CPU/GPU.
	 */
	static pyTensor* Alloc( uint32_t width, uint32_t height, uint32_t depth=1 );


	// tensor objects
	THFloatTensor* cpuTensor;
	THCudaTensor*  gpuTensor;	// (THCudaTensor defined as THCudaFloatTensor in THCGenerateAllTypes)

	PyObject* pyTensorCPU;	// Python handle to cpuTensor
	PyObject* pyTensorGPU;	// Python handle to gpuTensor

	float* cpuPtr;
	float* gpuPtr;

	uint32_t width;
	uint32_t height;
	uint32_t depth;

	size_t elements;
	size_t size;
};


typedef pyTensor Tensor;


#endif
#endif
