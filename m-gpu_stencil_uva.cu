// This is a 2D stencil example for multi-GPU using CUDA and only UVA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cassert>
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
    const int ngpus = 2;
    //cudaGetDeviceCount(&ngpus);
    printf("ngpus %i\n", ngpus);
    //assert(ngpus != 4);
    int* gpus = new int[ngpus];
    gpus[0] = 0; gpus[1] = 1;
    // Allocate memory for the grid on both GPUs using cudaMallocManaged()
    int **grid_dev = new int*[ngpus];
    // allocate using malloc managed
    // just divide the allocated memory by the nr of gpus
    int gridsize = (int)ceil(width*height/ngpus);
    // set device id in array gpus 
    // allocation to UVA for each GPU
    for(int i = 0; i < ngpus; i++){
	    printf("gpu[%i] set to %i\n", i,i);
	    cudaMallocManaged(&grid_dev[i], gridsize*sizeof(int));
    }
    // Initialize the grid values on host
    for(int i = 0; i < ngpus; i++){
	for (int j = 0; j < gridsize; j++) 
	{
            grid_dev[i][j] = 1;
	}
    }
     

    // prefetching to device
    for(int i = 0; i < ngpus; i++){
        cudaSetDevice(gpus[i]);
        cudaMemPrefetchAsync(grid_dev[i], gridsize*sizeof(int), gpus[i], NULL);
    }

    int is_able;
    cudaDeviceCanAccessPeer(&is_able, gpus[1], gpus[0]);
    if(is_able){
        cudaDeviceEnablePeerAccess(gpus[0], 0);
    }
    printf("Can GPU %i access GPU %i? %i \n", gpus[0], gpus[1], is_able);
    cudaSetDevice(gpus[0]);
    cudaDeviceCanAccessPeer(&is_able, gpus[0], gpus[1]);
    if(is_able){
        cudaDeviceEnablePeerAccess(gpus[1], 0);
    }
    printf("Can GPU %i access GPU %i? %i \n", gpus[1], gpus[0], is_able);
    /*
    for(int i = 0; i < ngpus; i++){
        cudaSetDevice(gpus[i]);
	// enable PtoP communication
	// check if GPU i can access GPU j
	for(int j = 0; i < ngpus; j++){
            if(gpus[i] != gpus[j]){
		cudaDeviceCanAccessPeer(&is_able, gpus[i], gpus[j]);
                printf("Can GPU %i access GPU %i? %i \n", gpus[j], gpus[i], is_able);
	    }
	    if(is_able){
	        cudaDeviceEnablePeerAccess(gpus[j], 0);
	    }
        }
    }
    */
    
    // Launch the kernel on both GPUs
    // threads per block
    dim3 block(block_size, block_size);
    // how many block fit in each grid
    // assuming the grid are equally large in each direction
    int nx = (int)ceil(width /block.x);
    int ny = (int)ceil(height/block.y);
    dim3 grid_size( nx, ny );
    int height_partition = int(height/ngpus);
    for (int iter = 0; iter < num_iterations; ++iter) {
        for(int i = 0; i < ngpus; i++){
	    // set device
	    cudaSetDevice(gpus[i]);
            // compute first half on GPU1
	    stencil_kernel<<<grid_size, block>>>(grid_dev[i], width, height_partition, 0, height_partition);
            
	    // synchronize with GPU i
	    for(int j = 0; i < ngpus; j++){
                if(i!=j){
		    cudaSetDevice(gpus[j]);
	            cudaDeviceSynchronize();
		}
            }
        }
    }
    // copy from GPU to print results on CPU using memcpy
    // here the GPUs have the full size grid and only
    // work on half of the domain
    int** grid_ngpu = new int*[gridsize]();
    for(int i = 0; i < ngpus; i++){
	// set device, is this necessary though?
        cudaSetDevice(gpus[i]);
	// copy from gpu i to host
        cudaMemcpy(grid_ngpu[i], grid_dev[i], gridsize*sizeof(int), cudaMemcpyDeviceToHost);
    }
    for(int i = 0; i < ngpus; i++){
        // Free the memory using cudaFree()
        cudaFree(grid_dev[i]);
    }

    // here GPUs have half sized grids
    // so results needs to be stiched together when copying from devices to host
    int size_grid  = int(width*height); 
    int *grid_full = new int[size_grid];
    int* ind_grid  = new int[ngpus]();
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
	    int ind = i*height + j;
	    for(int k=0; k <= ngpus; k++){    
                if (k*height_partition < j && j < (k+1)*height_partition){
		    printf("%i", ind_grid[k]);
		    grid_full[ind] = grid_ngpu[k][ind_grid[k]];
		    ind_grid[k] += 1;
		}
	    }
        }
    }

    // print results
    printf("Printing the output of the 2D stencil example\n");
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
	{
		std::cout << grid_full[i*height + j] << " ";
	}
	std::cout << std::endl;
    }
    return 0;
}

