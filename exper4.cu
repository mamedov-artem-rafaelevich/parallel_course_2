#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <nvtx3/nvToolsExt.h>
//#include <cub/device/device_radix_sort.h>

int main()
{
    cudaError_t stat;
    cudaSetDevice(3);
    float* arr=(float*)malloc(10*sizeof(float));
    float* cuarr;
    stat=cudaMalloc((void**)&cuarr,10*sizeof(float));
    if(stat!=cudaSuccess)
        printf("err 1: %d\n", stat);

    for(int i=0; i<10; i++)
        arr[i]=(float)i;

    stat=cudaMemcpy(cuarr,arr,10*sizeof(float),cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)
        printf("err 2: %d\n", stat);

    float* m;
    float ans;
    cudaMalloc(&m,sizeof(float));
    float* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    stat=cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, cuarr, m, 10);
    if(stat!=cudaSuccess)
        printf("err 3: %d\n", stat);

    // Allocate temporary storage
    stat=cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    if(stat!=cudaSuccess)
        printf("err 4: %d\n", stat);

    // Run max-reduction
    stat=cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, cuarr, m, 10);
    if(stat!=cudaSuccess)
        printf("err 5: %d\n", stat);
    cudaMemcpy((void*)&ans,(void*)m,sizeof(float),cudaMemcpyDeviceToHost);
    printf("%f\n",ans);

    // stat=cudaMemcpy((void*)arr,(void*)cuarr,10*sizeof(float),cudaMemcpyDeviceToHost);
    // if(stat!=cudaSuccess)
    //     printf("err 6: %d\n", stat);

    // stat=cudaFree((void*)cuarr);
    // if(stat!=cudaSuccess)
    //     printf("err 7: %d\n", stat);

    // stat=cudaFree((void*)d_temp_storage);
    // if(stat!=cudaSuccess)
    //     printf("err 8: %d\n", stat);

    free(arr);
    return 0;
}