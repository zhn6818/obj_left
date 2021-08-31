//
// Created by zhn68 on 2021/8/31.
//

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "test.h"

void print_array(int * array, int size)
{
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
}
__global__ void increment_atomic(int * g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i % array_size;
    atomicAdd(&g[i], 1);
}
void incres()
{
    printf("%d total threads in %d blocks writing into %d arrays\n", num_threads, num_threads / block_width, array_size);

    int h_array[array_size];
    const int array_bytes = array_size * sizeof(int);

    int * d_array;
    cudaMalloc((void **)&d_array, array_bytes);
    cudaMemset((void *)d_array, 0, array_bytes);

//    timer.Start();
    increment_atomic << <num_threads / block_width, block_width >> >(d_array);
//    timer.Stop();

    cudaMemcpy(h_array, d_array, array_bytes, cudaMemcpyDeviceToHost);
    print_array(h_array, array_size);
//    printf("\nTime elapsed = %g ms\n", timer.Elapsed());
    cudaFree(d_array);
}