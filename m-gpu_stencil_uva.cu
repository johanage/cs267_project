// This is a 2D stencil example for multi-GPU using CUDA and only UVA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cassert>
#include <omp.h>

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

__global__ void copy_gz_kernel_dest_right(int *destination, int* source, int width, int height, int gz_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sh = height-gz_height-1;
    if (i > 0 && i < width-1 && j > sh && j < height-1)
    {
        // Copy form source to dest
        destination[i*height + j - sh + gz_height] = source[i*height + j];
    }
}

__global__ void copy_gz_kernel_dest_left(int *destination, int* source, int width, int height, int gz_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sh = height-gz_height;
    if (i > 0 && i < width-1 && j > sh && j < height-1)
    {
        // Copy form source to dest
        destination[i*height + j] = source[i*height + j - sh + gz_height];
    }
}


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

int main(int argc, char *argv[]) {
    /*
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }*/

     // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-ngpus     <int>: set number of GPUs" << std::endl;
        std::cout << "-width     <int>: set width  of the 2D matrix" << std::endl;
        std::cout << "-height    <int>: set height of the 2D matrix" << std::endl;
        std::cout << "-print_res <int>: (0 or 1) print results" << std::endl;
        return 0;
    }


    const int width  = find_int_arg(argc, argv, "-width", 16);
    const int height = find_int_arg(argc, argv, "-height", 16);
    const int ngpus = find_int_arg(argc, argv, "-ngpus", 4);
    const int print_res = find_int_arg(argc, argv, "-v", 0);
    const int num_iterations = 1;
    const int block_size = 4;
    const int gz_height = 1;
    printf("ngpus %i\n", ngpus);
    int* gpus = new int[ngpus];
    for(int i = 0; i < ngpus; i++){
        gpus[i] = i;
    }
    // Allocate memory for the grid on both GPUs using cudaMallocManaged()
    int **grid_dev = new int*[ngpus]();
    // allocate using malloc managed
    // just divide the allocated memory by the nr of gpus
    int height_partition = int(height/ngpus);
    int height_partition_last = height - (ngpus-1)*height_partition;
    int gridsize = width + width*int(height/ngpus);
    int gridsizes[ngpus];
    int pad_bulk;
    for(int i = 0; i < ngpus; i++){
        if(ngpus%2 != 0 && i == ngpus-1)
	    {
                gridsizes[i] = width*height - (ngpus-1)*gridsize;
	    }
	    else{
                pad_bulk = int( i > 0 && i < ngpus-1);
	        gridsizes[i] = gridsize + pad_bulk*width;
	    }
    }
    
    // create streams to make event recording easier to handle
    cudaStream_t streams[ngpus];
    for(int i = 0; i < ngpus; i++){
        // create stream for gpu i
        cudaStreamCreate( &streams[i]);
    }
    
     
    // set device id in array gpus 
    // allocation to UVA for each GPU
    for(int i = 0; i < ngpus; i++){
        cudaMallocManaged(&grid_dev[i], gridsizes[i]*sizeof(int));
    }
    // Initialize the grid values on host
    for(int i = 0; i < ngpus; i++){
        for (int j = 0; j < gridsizes[i]; j++) 
        {
            if (j < width*int(height/ngpus) ){
		pad_bulk = int( i > 0 && i < ngpus-1);
	        grid_dev[i][j+pad_bulk*width] = 1;
	    }
        }
        // event experiment to measure the init memory transfer
        cudaEvent_t begin_mem, end_mem;
        cudaEventCreate(&begin_mem);
        cudaEventCreate(&end_mem);
        // start recording
        cudaEventRecord(begin_mem);
        
	// prefetching to device
        cudaSetDevice(gpus[i]);
        cudaMemPrefetchAsync(grid_dev[i], gridsizes[i]*sizeof(int), gpus[i], streams[i]);
        
	// end recording
        cudaEventRecord(end_mem);
        // synchronize
        cudaEventSynchronize(end_mem);
        float runtime_init_mem;
        cudaEventElapsedTime(&runtime_init_mem, begin_mem, end_mem);
        printf("The inital memory allocation and assignment took %f ms\n", runtime_init_mem);
        cudaEventDestroy(end_mem);
        cudaEventDestroy(begin_mem);
    }
    // print init
    // Launch the kernel on both GPUs
    // threads per block
    dim3 block(block_size, block_size);
    // how many block fit in each grid
    // assuming the grid are equally large in each direction
    int nx = (int)ceil(width /block.x);
    int ny = (int)ceil(height/block.y);
    dim3 grid_size( nx, ny );
    printf("Printing the initialzation of the 2D stencil example\n");
    for(int i = 0; i < ngpus; i++)
    {
	if(i < ngpus-1)
	{
            copy_gz_kernel_dest_left<<<grid_size, block>>>(grid_dev[i], grid_dev[i+1], width, height + width*gz_height, gz_height);
	    if(i > 0)
	    {
                copy_gz_kernel_dest_right<<<grid_size, block>>>(grid_dev[i], grid_dev[i-1], width, height + width*gz_height, gz_height);
	    }
	}
	cudaDeviceSynchronize();
    }
    for(int i = 0; i < ngpus; i++)
    {
	for(int j = 0; j < gridsizes[i]; j++)
            {
                std::cout << grid_dev[i][j] << " ";
		if(j == gridsizes[i] - width*gz_height-1 || i > 0 && i < ngpus-1 && j == width*gz_height-1){
		    std::cout << " | ";
		}

            }
            std::cout << std::endl;
    }
    // check and enable P2P access
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
    for (int iter = 0; iter < num_iterations; ++iter) {
#pragma omp parallel for
        for(int i = 0; i < ngpus; i++){
	    // start recording
            cudaEventRecord(begin);
	    // set device
            cudaSetDevice(gpus[i]);
            // copy GPU(i+1) GZ into GPU(i-1)
	    // corr gz on right side of GPUi and left part of GPUi-1
	    if (i < ngpus-1){
	        copy_gz_kernel_dest_right<<<grid_size, block>>>(grid_dev[i], grid_dev[i+1], width, height, gz_height);
                if(i > 0)
                {
                    copy_gz_kernel_dest_left<<<grid_size, block>>>(grid_dev[i], grid_dev[i-1], width, height, gz_height);
                }
	    }
	    // compute first half on GPU1
            if(i==ngpus-1 && ngpus%2 != 0){
	        stencil_kernel<<<grid_size, block>>>(grid_dev[i], width, height_partition, 0, height_partition_last-1);
	    }
	    else
	    {
		int x = int(i == ngpus-1);
                stencil_kernel<<<grid_size, block>>>(grid_dev[i], width, height_partition, 0, height_partition - x);
            }
	    // synchronize with GPU i
            for(int j = 0; j < ngpus; j++){
                if(i!=j){
                    cudaSetDevice(gpus[j]);
                    cudaDeviceSynchronize();
                }
            }
            // end recording
            cudaEventRecord(end);
            // synchronize
            cudaEventSynchronize(end);
            float runtime_computation;
            cudaEventElapsedTime(&runtime_computation, begin, end);
            //runtime_computation /= 1000; // given in ms
            printf("The kernel computation took %f ms\n", runtime_computation);
            cudaEventDestroy(end);
            cudaEventDestroy(begin);
        }
        
    }
    // copy from GPU to print results on CPU using memcpy
    // here the GPUs have the full size grid and only
    // work on half of the domain
    int** grid_ngpu = new int*[ngpus]();
    gridsize = width*int(height/ngpus);
    for(int i = 0; i < ngpus; i++){
        if(ngpus%2 != 0 && i == ngpus-1)
            {
                gridsizes[i] = width*height - (ngpus-1)*gridsize;
            }
            else{
                gridsizes[i] = gridsize;
            }
    } 
    for(int i = 0; i < ngpus; i++){
        // set device, is this necessary though?
        cudaSetDevice(gpus[i]);
        // event experiment to measure the init memory transfer
        cudaEvent_t begin_mem2, end_mem2;
        cudaEventCreate(&begin_mem2);
        cudaEventCreate(&end_mem2); 
        // start recording
        cudaEventRecord(begin_mem2);
	
	// allocate host mem
        grid_ngpu[i] = new int[gridsizes[i]]();
        // copy from gpu i to host
        pad_bulk = int( i > 0 && i < ngpus-1);
	// only copy bulk for array with structure # GZ # BULK # GZ #
	if(pad_bulk){
	    cudaMemcpyAsync(grid_ngpu[i], grid_dev[i] + width*gz_height, gridsizes[i]*sizeof(int), cudaMemcpyDefault, streams[i]);
	}
	// only copy left part for arr with structure # LP # GZ #
        else{
            cudaMemcpyAsync(grid_ngpu[i], grid_dev[i], gridsizes[i]*sizeof(int), cudaMemcpyDefault, streams[i]);
	}
	// Free the memory using cudaFree()
        cudaFree(grid_dev[i]);
	
	// end recording
        cudaEventRecord(end_mem2);
        // synchronize
        cudaEventSynchronize(end_mem2);
        float runtime_mem2;
        cudaEventElapsedTime(&runtime_mem2, begin_mem2, end_mem2);
        printf("Copying from device %i to host took %f ms\n", i, runtime_mem2);
        cudaEventDestroy(end_mem2);
        cudaEventDestroy(begin_mem2);
    }

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
                
		if(ngpus%2 != 0 && k == ngpus)
		{
                    if (height-height_partition_last <= j){
		        grid_full[ind] = grid_ngpu[k][ind_grid[k]];
		        ind_grid[k] += 1;
		    }
		}
		else
		{
                    if (k*height_partition <= j && j < (k+1)*height_partition){
		        grid_full[ind] = grid_ngpu[k][ind_grid[k]];
		        ind_grid[k] += 1;
		    }
                }
            }
        }
    }

    // print results
    if(print_res){
        printf("Printing the output of the 2D stencil example\n");
        for(int i = 0; i < width; i++)
        {
            for(int j = 0; j < height; j++)
	    {
	        std::cout << grid_full[i*height + j] << " ";
	    }
	    std::cout << std::endl;
        }
    }
    return 0;
}

