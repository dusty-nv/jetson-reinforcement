/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */
 
#include "cudaUtility.h"



// Convert an image (uchar3) from being interleaved by pixel, to planar band-sequential FP32 BGR
__global__ void gpuPackedToPlanarBGR( float2 scale, uchar3* input, int iWidth, float* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const uchar3 in  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(in.z, in.y, in.x);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
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

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPackedToPlanarBGR<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}

