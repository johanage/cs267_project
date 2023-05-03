// This is a 2D stencil example for multi-GPU using CUDA and only UVA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#define GPU1 0
#define GPU2 1

__global__ void stencil_kernel(float *grid, int width, int height, int sw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > sw && i < width-1 && j > 0 && j < height-1)
    {
        // Compute the new value of the grid point (i, j)
        float new_value = (grid[(i-1)*height+j] + grid[(i+1)*height+j]
                            + grid[i*height+j-1] + grid[i*height+j+1]);
        // Write the new value back to the grid
        grid[i*height+j] += new_value;
    }
}

int main() {
    const int width  = 8;
    const int height = 16;
    const int num_iterations = 1;
    const int block_size = 4;
    
    // Allocate memory for the grid on both GPUs using cudaMallocManaged()
    float *grid1, *grid2;
    // allocate using malloc managed
    int gridsize = (int)ceil(width*height/2); // remove div. by 2
    cudaMallocManaged(&grid1, gridsize*sizeof(int));
    cudaMallocManaged(&grid2, gridsize*sizeof(int));// remove div. by 2;
    
    // Initialize the grid values on GPU 1
    //printf(" Init grid1: \n");
    for (int i = 0; i < gridsize; i++) 
    {
	grid1[i] = 1;
	grid2[i] = 1;
    }
    
    // Launch the kernel on both GPUs
    // threads per block
    dim3 block(block_size, block_size);
    int nx = (int)ceil(width /block.x);
    int ny = (int)ceil(height/block.y);
    dim3 grid1_size( nx, ny );
    dim3 grid2_size( nx, ny );

    // prefetching to device
    cudaSetDevice(GPU1);
    cudaMemPrefetchAsync(grid1, gridsize*sizeof(int), GPU1, NULL);
    cudaSetDevice(GPU2);
    cudaMemPrefetchAsync(grid2, gridsize*sizeof(int), GPU2, NULL);

    // enable PtoP communication
    int is_able;
    // check if GPU2 can access GPU1
    cudaDeviceCanAccessPeer(&is_able, GPU2, GPU1);
    if(is_able){
        cudaDeviceEnablePeerAccess(GPU1, 0);
    }
    // check if GPU1 can access GPU2 if true enable access
    cudaDeviceCanAccessPeer(&is_able, GPU1, GPU2);
    cudaSetDevice(GPU1);
    if(is_able){
        cudaDeviceEnablePeerAccess(GPU2, 0);
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        // compute first half on GPU1 
	//stencil_kernel<<<grid1_size, block>>>(grid1, width, int(height/2), 0);
	stencil_kernel<<<grid1_size, block>>>(grid1, width, int(height/2), 0);
	//cudaSetDevice(GPU2);
        // synchronize with GPU2
	//cudaDeviceSynchronize();
	// compute other half on GPU2
	//stencil_kernel<<<grid2_size, block>>>(grid2,int(width/2), height, 0);
        //cudaDeviceSynchronize();
    }
    // copy from GPU to print results on CPU using memcpy
    // here the GPUs have the full size grid and only
    // work on half of the domain
    float *grid_1 = new float[gridsize]();
    float *grid_2 = new float[gridsize]();
    //cudaMemcpy(grid_2, grid2, gridsize*sizeof(int), cudaMemcpyDeviceToHost);
    // was on GPU2 
    //cudaSetDevice(GPU1);
    cudaMemcpy(grid_1, grid1, gridsize*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free the memory using cudaFree()
    cudaFree(grid1);
    cudaFree(grid2);

    // here GPUs have half sized grids
    // so results needs to be stiched together when copying from devices to host
    int size_grid = int(width*height); 
    int *grid = new int[size_grid];
    int ind_grid1 = 0;
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
	    int ind = i*int(height/2) + j;
	    if (j < int(height/2)){
	        printf(" index %i", ind);
		grid[ind] = grid_1[ind_grid1];
		ind_grid1 += 1;
	    }
	    //grid[i] = grid_2[i];
	    else{
                grid[ind] = 0;
	    }
        }
    }

    // print results
    printf("Printing the output of the 2D stencil example\n");
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
	{
		//std::cout << i*width + j << " : " << grid_2[i*width + j] << " "; // << std::endl;
		//std::cout << grid_1[i*width + j] << " "; // << std::endl;
		//std::cout << grid_2[i*width + j] << " "; // << std::endl;
		std::cout << grid[i*width + j] << " "; // << std::endl;
	}
	std::cout << std::endl;
    }
    return 0;
}

