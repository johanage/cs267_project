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
    const int width  = 2048;
    const int height = 2048;
    const int num_iterations = 1;
    const int block_size = 4;
    const int ngpus = 4;
    printf("ngpus %i\n", ngpus);
    int* gpus = new int[ngpus];
    for(int i = 0; i < ngpus; i++){
	    gpus[i] = i;
    }
    //gpus[0] = 0; gpus[1] = 1;
    // Allocate memory for the grid on both GPUs using cudaMallocManaged()
    int **grid_dev = new int*[ngpus];
    // allocate using malloc managed
    // just divide the allocated memory by the nr of gpus
    int gridsize = (int)ceil(width*height/ngpus);
    
    // event experiment to measure the init memory transfer
    cudaEvent_t begin_mem, end_mem;
    cudaEventCreate(&begin_mem);
    cudaEventCreate(&end_mem);
    
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
    cudaEventRecord(end_mem);
    cudaEventSynchronize(end_mem);
    float runtime_init_mem;
    cudaEventElapsedTime(&runtime_init_mem, begin_mem, end_mem);
    printf("The inital memory allocation and assignment took %f ms\n", runtime_init_mem);
    cudaEventDestroy(end_mem);
    cudaEventDestroy(begin_mem);


    int is_able;
    for(int i = 0; i < ngpus; i++){
        cudaSetDevice(gpus[i]);
	// enable PtoP communication
	// check if GPU i can access GPU j
	for(int j = 0; j < ngpus; j++){
            if(gpus[i] != gpus[j]){
		cudaDeviceCanAccessPeer(&is_able, gpus[i], gpus[j]);
                printf("Can GPU %i access GPU %i? %i \n", gpus[j], gpus[i], is_able);
	    }
	    if(is_able){
	        cudaDeviceEnablePeerAccess(gpus[j], 0);
	    }
        }
    }
    
    // event experiment to measure computation time
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end); 
    
    // Launch the kernel on both GPUs
    // threads per block
    dim3 block(block_size, block_size);
    // how many block fit in each grid
    // assuming the grid are equally large in each direction
    int nx = (int)ceil(width /block.x);
    int ny = (int)ceil(height/block.y);
    dim3 grid_size( nx, ny );
    int height_partition = int(height/ngpus);
    cudaEventRecord(begin);
    for (int iter = 0; iter < num_iterations; ++iter) {
        for(int i = 0; i < ngpus; i++){
	    // set device
	    cudaSetDevice(gpus[i]);
            // compute first half on GPU1
	    if(i==0){
	        stencil_kernel<<<grid_size, block>>>(grid_dev[i], width, height_partition, 0, height_partition);
	    }
	    else if(i==ngpus-1){
	        stencil_kernel<<<grid_size, block>>>(grid_dev[i], width, height_partition, -1, height_partition-1);
	    }
	    else{
	        stencil_kernel<<<grid_size, block>>>(grid_dev[i], width, height_partition, -1, height_partition);
            }
	    // synchronize with GPU i
	    for(int j = 0; j < ngpus; j++){
                if(i!=j){
		    cudaSetDevice(gpus[j]);
	            cudaDeviceSynchronize();
		}
            }
        }
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float runtime_computation;
    cudaEventElapsedTime(&runtime_computation, begin, end);
    //runtime_computation /= 1000; // given in ms
    printf("The kernel computation took %f ms\n", runtime_computation);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
   

    // event experiment to measure the init memory transfer
    cudaEvent_t begin_mem2, end_mem2;
    cudaEventCreate(&begin_mem2);
    cudaEventCreate(&end_mem2);
 
    
    // copy from GPU to print results on CPU using memcpy
    // here the GPUs have the full size grid and only
    // work on half of the domain
    int** grid_ngpu = new int*[ngpus]();
    for(int i = 0; i < ngpus; i++){
	// set device, is this necessary though?
        cudaSetDevice(gpus[i]);
        // allocate host mem
	grid_ngpu[i] = new int[gridsize]();
	// copy from gpu i to host
        cudaMemcpy(grid_ngpu[i], grid_dev[i], gridsize*sizeof(int), cudaMemcpyDefault);
    }
    for(int i = 0; i < ngpus; i++){
        // Free the memory using cudaFree()
        cudaFree(grid_dev[i]);
    }

    cudaEventRecord(end_mem2);
    cudaEventSynchronize(end_mem2);
    float runtime_mem2;
    cudaEventElapsedTime(&runtime_mem2, begin_mem2, end_mem2);
    printf("Copying from devices to host took %f ms\n", runtime_mem2);
    cudaEventDestroy(end_mem2);
    cudaEventDestroy(begin_mem2);

    // here GPUs have half sized grids
    // so results needs to be stiched together when copying from devices to host
    int size_grid  = int(width*height); 
    int *grid_full = new int[size_grid]();
    int* ind_grid  = new int[ngpus]();
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
	    int ind = i*height + j;
	    for(int k=0; k <= ngpus; k++){    
                if (k*height_partition <= j && j < (k+1)*height_partition){
		    //printf("%i", ind_grid[k]);
		    grid_full[ind] = grid_ngpu[k][ind_grid[k]];
		    ind_grid[k] += 1;
		}
	    }
        }
    }

    // print results
    /*
    printf("Printing the output of the 2D stencil example\n");
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
	{
		std::cout << grid_full[i*height + j] << " ";
	}
	std::cout << std::endl;
    }
    */
    return 0;
}

