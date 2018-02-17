/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __CUDA_PLANAR_IMAGE_H_
#define __CUDA_PLANAR_IMAGE_H_


#include "cudaUtility.h"

							   
/**
 * Convert a packed uchar3 image to a floating-point planar BGR image (band-sequential)
 */
cudaError_t cudaPackedToPlanarBGR( uchar3* input, size_t inputWidth, size_t inputHeight,
				                   float* output, size_t outputWidth, size_t outputHeight );

/**
 * Convert a packed float4 image to a floating-point planar BGR image (band-sequential)
 */
cudaError_t cudaRGBAToPlanarBGR( float4* input, size_t inputWidth, size_t inputHeight, 
				                 float* output, size_t outputWidth, size_t outputHeight );

								 
/**
 * Convert a packed float4 image to a floating-point planar BGR image (band-sequential)
 */
cudaError_t cudaRGBAToPlanarBGR( float4* input, size_t inputWidth, size_t inputHeight, const float2& inputRange,
				                 float* output, size_t outputWidth, size_t outputHeight, const float2& outputRange );

								 
#endif
