#include <cuda.h>
#include <iostream>
#include <random>

#define radius 2
#define NUM_THREADS 128

// cpu variables
int* stencil;
int N;

// device variables
int* d_stencil;
int* d_N;

__global__ void init_variables(int* x)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= N){
		return;
        }
	x[tid] = 1;
	d_N = N;
}

__global__ void init_computation(int n)
{
	N  = n;
	// alloc size variable to GPU
	cudaMalloc((void**)&d_N, sizeof(int));
	cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
	// alloc stencil to GPU
	cudaMalloc((void**)&d_stencil, N);
	cudaMemcpy(d_stencil, &stencil, N, cudaMemcpyHostToDevice);
}

__global__ void compute_stencil(int* x)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= N){
                return;
        }
	
}
