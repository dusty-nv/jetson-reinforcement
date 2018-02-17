/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */
 
#include "cudaUtility.h"



// Convert an image (uchar3) from being interleaved by pixel, to planar band-sequential FP32 BGR
template <typename T> __global__ void gpuPackedToPlanarBGR( T* input, int iWidth, float* output, int oWidth, int oHeight,
														    float2 scale, float range_multiplier )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T in       = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(in.z, in.y, in.x);
	
	output[n * 0 + y * oWidth + x] = bgr.x * range_multiplier;
	output[n * 1 + y * oWidth + x] = bgr.y * range_multiplier;
	output[n * 2 + y * oWidth + x] = bgr.z * range_multiplier;
}


// cudaPackedToPlanarBGR
cudaError_t cudaPackedToPlanarBGR( uchar3* input, size_t inputWidth, size_t inputHeight,
				               float* output, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float range_multiplier = 1.0f;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPackedToPlanarBGR<uchar3><<<gridDim, blockDim>>>(input, inputWidth, output, outputWidth, outputHeight, scale, range_multiplier);

	return CUDA(cudaGetLastError());
}


// cudaRGBAToPlanarBGR
cudaError_t cudaRGBAToPlanarBGR( float4* input, size_t inputWidth, size_t inputHeight, const float2& inputRange,
				                 float* output, size_t outputWidth, size_t outputHeight, const float2& outputRange )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float range_multiplier = outputRange.y / inputRange.y;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPackedToPlanarBGR<float4><<<gridDim, blockDim>>>(input, inputWidth, output, outputWidth, outputHeight, scale, range_multiplier);

	return CUDA(cudaGetLastError());
}


// cudaRGBAToPlanarBGR
cudaError_t cudaRGBAToPlanarBGR( float4* input, size_t inputWidth, size_t inputHeight,
				                 float* output, size_t outputWidth, size_t outputHeight )
{
	return cudaRGBAToPlanarBGR(input, inputWidth, inputHeight, make_float2(0.0f, 1.0f),
							   output, outputWidth, outputHeight, make_float2(0.0f, 1.0f));
}

