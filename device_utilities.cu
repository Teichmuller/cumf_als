#include "device_utilities.h"

void saveDeviceFloatArrayToFile(const std::string& fileName, int size, dtype* d_array)
{
	dtype* h_array;
	cudacall(cudaMallocHost( (void** ) &h_array, size * sizeof(h_array[0])) );
	cudacall(cudaMemcpy(h_array, d_array, size * sizeof(h_array[0]),cudaMemcpyDeviceToHost));
	FILE * outfile = fopen(fileName.data(), "wb");
	fwrite(h_array, sizeof(dtype), size, outfile);
	fclose(outfile);
	cudaFreeHost(h_array);
}

__global__ void fp32Array2fp16Array(const float * fp32Array, half* fp16Array,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		fp16Array[i] =  __float2half(fp32Array[i]);
	}
}

__global__ void fp16Array2fp32Array(float * fp32Array, const half* fp16Array,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		fp32Array[i] =  __half2float(fp16Array[i]);
	}
}

