#include <stdio.h>
#include <cuda_runtime.h>

// Define the kernel function for stencil operation
__global__ void stencil(int* grid, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        grid[i * N + j] += (grid[(i - 1) * N + j] + grid[(i + 1) * N + j] + grid[i * N + j - 1] + grid[i * N + j + 1]);
    }
}

int main()
{
    int num_gpus = 4; // Number of GPUs to use
    int N = 32; // Size of the grid
    int size = N * N * sizeof(float); // Size of grid in bytes
    int threads_per_block = 4; // Number of threads per block
    int blocks_per_dim = N / threads_per_block; // Number of blocks per dimension

    // Initialize the grid on the CPU
    int* grid = new int[N * N];
    for (int i = 0; i < N * N; i++) {
        grid[i] = 1;
    }

    // Declare arrays to hold device pointers and GPU IDs
    int** grid_dev = new int*[num_gpus];
    int* gpus = new int[num_gpus];

    // Initialize the GPUs and device pointers
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaMalloc(&grid_dev[i], size / num_gpus);
        cudaMemcpy(&grid_dev[i][(N / num_gpus) * N], &grid[(N / num_gpus) * N * i], size / num_gpus, cudaMemcpyHostToDevice);
        gpus[i] = i;
    }

    // Enable peer-to-peer access between GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(gpus[i]);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                cudaDeviceEnablePeerAccess(gpus[j], gpus[i]);
            }
        }
    }

    // Launch the kernel on each GPU
    dim3 threads(threads_per_block, threads_per_block);
    for (int iter = 0; iter < 1; iter++) {
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(gpus[i]);
            dim3 blocks(blocks_per_dim / num_gpus, blocks_per_dim);
            stencil<<<blocks, threads>>>(grid_dev[i], N / num_gpus);

            // Synchronize with other GPUs
            for (int j = 0; j < num_gpus; j++) {
                if (i != j) {
                    cudaSetDevice(gpus[j]);
                    cudaDeviceSynchronize();
                }
            }
        }
    }

    // Copy the results back to the CPU and print them out
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(gpus[i]);
        cudaMemcpy(&grid[(N / num_gpus) * N * i], &grid_dev[i][(N / num_gpus) * N], size / num_gpus, cudaMemcpyDeviceToHost);
    }
    // Print the results
    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%i ", grid[i * N + j]);
        }
        printf("\n");
    }

    // Free the memory
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(gpus[i]);
        cudaFree(grid_dev[i]);
        cudaDeviceReset();
    }
    delete[] grid;
    delete[] grid_dev;
    delete[] gpus;

    return 0;
}
