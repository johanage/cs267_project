#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

float p2p_copy (size_t size)
{
	int *pointers[2];

	cudaSetDevice (0);
	cudaDeviceEnablePeerAccess (1, 0);
	cudaMalloc (&pointers[0], size);

	cudaSetDevice (1);
	cudaDeviceEnablePeerAccess (0, 0);
	cudaMalloc (&pointers[1], size);

	cudaEvent_t begin, end;
	cudaEventCreate (&begin);
	cudaEventCreate (&end);

	cudaEventRecord (begin);
	cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
	cudaEventRecord (end);
	cudaEventSynchronize (end);

	float elapsed;
	cudaEventElapsedTime (&elapsed, begin, end);
	elapsed /= 1000;

	cudaSetDevice (0);
	cudaFree (pointers[0]);

	cudaSetDevice (1);
	cudaFree (pointers[1]);

	cudaEventDestroy (end);
	cudaEventDestroy (begin);

	return elapsed;
}

int main(){
	float elapse_p2p_copy;
	size_t size = 100000;
	elapse_p2p_copy = p2p_copy(size);
	printf("Elapsed time p2p copy: %f s \n", elapse_p2p_copy);
}
