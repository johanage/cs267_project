// This is a 2D stencil example for multi-GPU using CUDA and only UVA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void stencil_kernel(float *grid, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < width - 1 && j > 0 && j < height - 1)
    {
        // Compute the new value of the grid point (i, j)
        float new_value = (grid[(i-1)*height+j] + grid[(i+1)*height+j]
                            + grid[i*height+j-1] + grid[i*height+j+1]);
        // Write the new value back to the grid
        grid[i*height+j] += new_value;
    }
}

int main() {
    const int width = 16;
    const int height = 16;
    const int num_iterations = 1;
    const int block_size = 4;
    
    // Allocate memory for the grid on both GPUs using cudaMallocManaged()
    float *grid1, *grid2;
    cudaMallocManaged(&grid1, width*height*sizeof(float));
    cudaMallocManaged(&grid2, width*height*sizeof(float));
    
    // Initialize the grid values on GPU 1
    //printf(" Init grid1: \n");
    int sum = 0;
    for (int i = 0; i < int( width*height ); i++) 
    {
        grid1[i] = 1;
	sum += grid1[i];
	std::cout << sum << " ";
    }
    
    // Launch the kernel on both GPUs
    // threads per block
    dim3 block(block_size, block_size);
    int nx = (int)ceil(width/block.x);
    int ny = (int)ceil(height/block.y);
    dim3 grid1_size( nx, ny );
    dim3 grid2_size( nx, ny );

    // prefetching to device
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(grid1, width*height*sizeof(int), device, NULL);
    cudaMemPrefetchAsync(grid2, width*height*sizeof(int), device, NULL);

    for (int iter = 0; iter < num_iterations; ++iter) {
        stencil_kernel<<<grid1_size, block>>>(grid1, width, height);
        stencil_kernel<<<grid2_size, block>>>(grid2, width, height);
        cudaDeviceSynchronize();
        // Swap the grids so that the updated values are on the other GPU for the next iteration
        float *temp = grid1;
        grid1 = grid2;
        grid2 = temp;
    }
    // copy from GPU to print results on CPU
    int size_grid = int(width*height); 
    printf("Size of grid %i ", size_grid);
    float *grid   = new float[size_grid]();
    float *grid_2 = new float[size_grid]();
    cudaMemcpy(grid,   grid1, size_grid*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_2, grid2, size_grid*sizeof(int), cudaMemcpyDeviceToHost);
    // Free the memory using cudaFree()
    cudaFree(grid1);
    cudaFree(grid2);
    printf("Printing the output of the 2D stencil example\n");
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
	{
		//std::cout << i*width + j << " : " << grid_2[i*width + j] << " "; // << std::endl;
		//std::cout << grid[i*width + j] << " "; // << std::endl;
		std::cout << grid_2[i*width + j] << " "; // << std::endl;
		//std::cout << grid[i*width + j] + grid_2[i*width + j] << " "; // << std::endl;
	}
	std::cout << std::endl;
    }
    return 0;
}

