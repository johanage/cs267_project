// This is a 2D stencil example for multi-GPU using CUDA and only UVA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#define GPU1 0
#define GPU2 1

__global__ void stencil_kernel(int *grid, int width, int height, int sh, int eh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < width-1 && j > sh && j < eh)
    {
        // Compute the new value of the grid point (i, j)
        int new_value = (grid[(i-1)*height+j] + grid[(i+1)*height+j]
                            + grid[i*height+j-1] + grid[i*height+j+1]);
        // Write the new value back to the grid
        grid[i*height+j] += new_value;
    }
}

int main() {
    const int width  = 16;
    const int height = 16;
    const int num_iterations = 1;
    const int block_size = 4;
    
    // Allocate memory for the grid on both GPUs using cudaMallocManaged()
    int *grid1, *grid2;
    // allocate using malloc managed
    int gridsize = (int)ceil(width*height/2); // remove div. by 2
    cudaMallocManaged(&grid1, gridsize*sizeof(int));
    cudaMallocManaged(&grid2, gridsize*sizeof(int));// remove div. by 2;
    
    // Initialize the grid values on GPU 1
    for (int i = 0; i < gridsize; i++) 
    {
	grid1[i] = 1;
	grid2[i] = 1;
    }
    // prefetching to device
    cudaSetDevice(GPU1);
    cudaMemPrefetchAsync(grid1, gridsize*sizeof(int), GPU1, NULL);
    cudaSetDevice(GPU2);
    cudaMemPrefetchAsync(grid2, gridsize*sizeof(int), GPU2, NULL);

    // enable PtoP communication
    int is_able;
    // check if GPU2 can access GPU1
    cudaDeviceCanAccessPeer(&is_able, GPU2, GPU1);
    printf("Can GPU2 access GPU1? %i \n", is_able);
    if(is_able){
        cudaDeviceEnablePeerAccess(GPU1, 0);
    }
    // check if GPU1 can access GPU2 if true enable access
    cudaSetDevice(GPU1);
    cudaDeviceCanAccessPeer(&is_able, GPU1, GPU2);
    printf("Can GPU1 access GPU2? %i \n", is_able);
    if(is_able){
        cudaDeviceEnablePeerAccess(GPU2, 0);
    }
    
    // Launch the kernel on both GPUs
    // threads per block
    dim3 block(block_size, block_size);
    // how many block fit in each grid
    // assuming the grid are equally large in each direction
    int nx = (int)ceil(width /block.x);
    int ny = (int)ceil(height/block.y);
    dim3 grid1_size( nx, ny );
    dim3 grid2_size( nx, ny );
    int half_height = int(height/2);
    for (int iter = 0; iter < num_iterations; ++iter) {
        // compute first half on GPU1
	stencil_kernel<<<grid1_size, block>>>(grid1, width, half_height, 0, half_height);
	cudaSetDevice(GPU2);
        // synchronize with GPU2
	cudaDeviceSynchronize();
	// compute other half on GPU2
	stencil_kernel<<<grid2_size, block>>>(grid2, width, half_height, -1, half_height-1);
        //cudaDeviceSynchronize();
    }
    // copy from GPU to print results on CPU using memcpy
    // here the GPUs have the full size grid and only
    // work on half of the domain
    int *grid_1 = new int[gridsize]();
    int *grid_2 = new int[gridsize]();
    cudaMemcpy(grid_2, grid2, gridsize*sizeof(int), cudaMemcpyDeviceToHost);
    // was on GPU2 
    cudaSetDevice(GPU1);
    cudaMemcpy(grid_1, grid1, gridsize*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free the memory using cudaFree()
    cudaFree(grid1);
    cudaFree(grid2);

    // here GPUs have half sized grids
    // so results needs to be stiched together when copying from devices to host
    int size_grid = int(width*height); 
    int *grid = new int[size_grid];
    int ind_grid1 = 0;
    int ind_grid2 = 0;
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
	    int ind = i*height + j;
	    if (j < int(height/2)){
		grid[ind] = grid_1[ind_grid1];
		ind_grid1 += 1;
	    }
	    else{
                grid[ind] = grid_2[ind_grid2];
		ind_grid2 += 1;
	    }
        }
    }

    // print results
    printf("Printing the output of the 2D stencil example\n");
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
	{
		//std::cout << i*height + j << " : " << grid_2[i*height + j] << " "; // << std::endl;
		//std::cout << grid_1[i*height + j] << " "; // << std::endl;
		//std::cout << grid_2[i*height + j] << " "; // << std::endl;
		std::cout << grid[i*height + j] << " "; // << std::endl;
	}
	std::cout << std::endl;
    }
    return 0;
}

