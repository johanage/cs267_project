// This is a stencil example for multi-GPU using CUDA + nvshmem
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
//#include <shmemx.h>

#define BLOCK_DIM 16
#define SHMEM_SIZE (BLOCK_DIM + 2)

// globals
dim3 dimGrid;
dim3 dimBlock;

__global__ void stencil_kernel(float* input, float* output, int width, int height)
{
    __shared__ float shared_mem[SHMEM_SIZE][SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_DIM;
    int by = blockIdx.y * BLOCK_DIM;

    int x = bx + tx;
    int y = by + ty;

    if (x < width && y < height)
    {
        // Load the shared memory
        shared_mem[tx+1][ty+1] = input[x + y * width];
        if (tx == 0 && bx > 0)
            shared_mem[0][ty+1] = input[(bx-1) + y * width];
        if (tx == BLOCK_DIM-1 && bx+BLOCK_DIM < width)
            shared_mem[SHMEM_SIZE-1][ty+1] = input[(bx+BLOCK_DIM) + y * width];
        if (ty == 0 && by > 0)
            shared_mem[tx+1][0] = input[x + (by-1) * width];
        if (ty == BLOCK_DIM-1 && by+BLOCK_DIM < height)
            shared_mem[tx+1][SHMEM_SIZE-1] = input[x + (by+BLOCK_DIM) * width];
        __syncthreads();
        // Compute the stencil
        if (x > 0 && y > 0 && x < width-1 && y < height-1)
        {
            float sum = 0;
            for (int i=-1; i<=1; i++)
                for (int j=-1; j<=1; j++)
                    sum += shared_mem[tx+1+i][ty+1+j];
            output[x + y * width] = sum / 9;
        }
        else
		{
            output[x + y * width] = input[x + y * width];
		}
    }
}

int main(int argc, char* argv[])
{
    int rank, size;
    cudaStream_t stream;
    float *d_input, *d_output;
    nvshmem_init();
    rank = nvshmem_my_pe();
    size = nvshmem_n_pes();

    if (rank == 0)
    {
        int width = 1024;
        int height = 1024;

        // Initialize input array
        float* input = (float*) malloc(width * height * sizeof(float));
        for (int i=0; i<width*height; i++)
		{
            input[i] = (float) i;
		}
        // Allocate device memory
        cudaMalloc((void**) &d_input, width * height * sizeof(float));
        cudaMalloc((void**) &d_output, width * height * sizeof(float));

        // Copy input to device memory
        cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel on all devices
        dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 dimGrid((width + BLOCK_DIM-1) / BLOCK_DIM, (height + BLOCK_DIM-1) / BLOCK_DIM);
        cudaStreamCreate(&stream);
        for (int i=1; i<size; i++)
        {
            nvshmem_barrier_all();
            cudaSetDevice(i);
            nvshmem_quiet();
			cudaMemcpyPeerAsync(d_input, i, d_input, 0, width * height * sizeof(float), stream);
			cudaMemcpyPeerAsync(d_output, i, d_output, 0, width * height * sizeof(float), stream);
			stencil_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_input, d_output, width, height);
			cudaMemcpyPeerAsync(input, 0, d_output, i, width * height * sizeof(float), stream);
			cudaStreamSynchronize(stream);
			cudaSetDevice(0);
			nvshmem_quiet();
		}
		// Compute the stencil on the first device
		stencil_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
		// Copy output to host memory
		cudaMemcpy(input, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
		// Free device memory
		cudaFree(d_input);
		cudaFree(d_output);
		// Print the result
		printf("Stencil result:\n");
		for (int i=0; i<10; i++)
		{
			for (int j=0; j<10; j++)
			{
				printf("%f ", input[i+j*width]);
			}
			printf("\n");
		}
		// Free host memory
		free(input);
	}
	else
	{
		cudaSetDevice(rank);
		cudaMalloc((void**) &d_input, 1024 * 1024 * sizeof(float));
		cudaMalloc((void**) &d_output, 1024 * 1024 * sizeof(float));
		nvshmem_barrier_all();
		nvshmem_quiet();
		stencil_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, 1024, 1024);
		cudaMemcpyPeerAsync(d_output, 0, d_output, rank, 1024 * 1024 * sizeof(float), stream);
		nvshmem_quiet();
		cudaFree(d_input);
		cudaFree(d_output);
	}

	nvshmem_finalize();
	return 0;
}
