#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "nccl.h"

#define NX 1024
#define NY 1024
#define NZ 1024
#define HALO 1

__global__ void stencil(float *in, float *out, int nx, int ny, int nz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= HALO && i < nx - HALO && j >= HALO && j < ny - HALO && k >= HALO && k < nz - HALO) {
        out[i * ny * nz + j * nz + k] = (in[(i - 1) * ny * nz + j * nz + k] +
                                         in[(i + 1) * ny * nz + j * nz + k] +
                                         in[i * ny * nz + (j - 1) * nz + k] +
                                         in[i * ny * nz + (j + 1) * nz + k] +
                                         in[i * ny * nz + j * nz + k - 1] +
                                         in[i * ny * nz + j * nz + k + 1]) / 6.0f;
    }
}

int main(int argc, char *argv[])
{
    int nGPUs = 2;
    int deviceIDs[2] = {0, 1};

    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclCommInitRank(&comm, nGPUs, id, 0, deviceIDs);

    // Allocate memory on each GPU
    cudaSetDevice(deviceIDs[0]);
    float *d_in0, *d_out0;
    cudaMalloc((void **)&d_in0, NX * NY * NZ * sizeof(float));
    cudaMalloc((void **)&d_out0, NX * NY * NZ * sizeof(float));

    cudaSetDevice(deviceIDs[1]);
    float *d_in1, *d_out1;
    cudaMalloc((void **)&d_in1, NX * NY * NZ * sizeof(float));
    cudaMalloc((void **)&d_out1, NX * NY * NZ * sizeof(float));

    // Initialize input data on the first GPU
    cudaSetDevice(deviceIDs[0]);
    float *h_in = (float *)malloc(NX * NY * NZ * sizeof(float));
    for (int i = 0; i < NX * NY * NZ; i++) {
        h_in[i] = i;
    }
    cudaMemcpy(d_in0, h_in, NX * NY * NZ * sizeof(float), cudaMemcpyHostToDevice);

    // Synchronize NCCL
    ncclCommCuDevice(comm, deviceIDs[0]);
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);
    ncclGroupStart();
    ncclSend(d_in0, NX * NY * NZ, ncclFloat, 0, comm, stream0);
    cudaSetDevice(deviceIDs[1]);
    float *d_in1_recv;
    cudaMalloc((void **)&d_in1_recv, NX * NY * NZ * sizeof(float));
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    ncclRecv(d_in1, NX * NY * NZ, ncclFloat, 0, comm, stream1);
	ncclGroupEnd();
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	// Perform the stencil operation on both GPUs
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((NX + blockDim.x - 1) / blockDim.x, (NY + blockDim.y - 1) / blockDim.y, (NZ + blockDim.z - 1) / blockDim.z);
	cudaSetDevice(deviceIDs[0]);
	stencil<<<gridDim, blockDim>>>(d_in0, d_out0, NX, NY, NZ);

	cudaSetDevice(deviceIDs[1]);
	stencil<<<gridDim, blockDim>>>(d_in1_recv, d_out1, NX, NY, NZ);

	// Synchronize NCCL
	ncclCommCuDevice(comm, deviceIDs[1]);
	cudaStream_t stream2;
	cudaStreamCreate(&stream2);
	ncclGroupStart();
	ncclSend(d_out1, NX * NY * NZ, ncclFloat, 1, comm, stream2);
	cudaSetDevice(deviceIDs[0]);
	float *d_out0_recv;
	cudaMalloc((void **)&d_out0_recv, NX * NY * NZ * sizeof(float));
	cudaStream_t stream3;
	cudaStreamCreate(&stream3);
	ncclRecv(d_out0_recv, NX * NY * NZ, ncclFloat, 1, comm, stream3);
	ncclGroupEnd();
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);

	// Combine the results on the first GPU
	cudaSetDevice(deviceIDs[0]);
	for (int i = HALO; i < NX - HALO; i++) {
		for (int j = HALO; j < NY - HALO; j++) {
			for (int k = HALO; k < NZ - HALO; k++) {
				h_in[i * NY * NZ + j * NZ + k] = (d_out0_recv[i * NY * NZ + j * NZ + k] + d_out0[i * NY * NZ + j * NZ + k]) / 2.0f;
			}
		}
	}

	// Free memory
	cudaFree(d_in0);
	cudaFree(d_out0);
	cudaFree(d_in1);
	cudaFree(d_out1);
	cudaFree(d_in1_recv);
	cudaFree(d_out0_recv);
	free(h_in);

	// Finalize NCCL
	ncclCommDestroy(comm);

	return 0;
}
