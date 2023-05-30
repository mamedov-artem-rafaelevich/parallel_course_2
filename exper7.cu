#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <cub/cub.cuh>

int main()
{
    cudaError_t stat;
    int  num_items;      // e.g., 7
    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  h_in[] = {8, 6, 7, 5, 3, 0, 9};
    int  *d_max;         // e.g., [-]
    stat = cudaMalloc((void**)&d_in,7*sizeof(int));
    if(stat!=cudaSuccess)
        printf("err 1: %d",stat);
    stat = cudaMalloc((void**)&d_max,sizeof(int));
    if(stat!=cudaSuccess)
        printf("err 2: %d",stat);
    stat=cudaMemcpy(d_in,h_in,7*sizeof(int),cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)
        printf("err 3: %d\n", stat);
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
    // d_out <-- [9]
    std::cout << d_max[0];
}