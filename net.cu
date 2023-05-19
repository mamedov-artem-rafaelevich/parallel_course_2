#include <math.h>
#include <cuda.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/10.2/include/cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

class Linear
{
    private:
    float* array;
    float* cuarray;
    int w,h;
    public:
    Linear();
    Linear(int a, int b)
    {
        this->array = (float*)malloc(a*sizeof(float));
        cudaMalloc(&this->cuarray,a*b);
        cudaMalloc(&this->cuout,b);
        this->h=a;
        for(int i=0; i<a; i++)
            for(int j=0; j<b; j++)
                this->array[IDX2C(i,j,this->h)]=float(rand()%255);
        cudaMemcpy(this->cuarray,this->array,a*b,cudaMemcpyHostToDevice);
    };
    ~Linear()
    {
        cudaMemcpy(this->cuarray,this->array,this->w*this->h,cudaMemcpyDeviceToHost);
        cudaFree(this->cuarray);
        cudaFree(this->cuout);
        free(array);
    };

    float* cuout;

    float* forward(float* arr,cublasHandle_t handle)
    {
        float* skal;
        cudaMalloc(&skal,this->h*this->w);
        for(int i=0; i<this->h*this->w; i++)
            skal[i]=1;
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,this->h,this->w,1,skal,arr,this->h,this->cuarray,this->w,skal,this->cuout,this->w);
        cudaFree(arr);
        return this->cuout;
    };
};

__device__ float sigma(float a)
{
    return (1/(1-exp(-a)));
}

__global__ void sigmoid(float* arr,int n)
{
//    int i=blockIdx.x*blockDim.x+threadIdx.x;
//    int j=blockIdx.y*blockDim.y+threadIdx.y;
    arr[IDX2C(threadIdx.x,blockIdx.x,n)]=sigma(arr[IDX2C(threadIdx.x,blockIdx.x,n)]);
}

class Net
{
    public:
    Linear fc1,fc2,fc3;
    
    Net()
    {
        printf("init\n");
        this->fc1=Linear(32*32,16*16);
        this->fc2=Linear(16*16,4*4);
        this->fc3=Linear(4*4,1);
    };

    float* forward(float* arr)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        dim3 a(32,32);
        dim3 b(16,16);
        dim3 c(4,4);
        arr=this->fc1.forward(arr,handle);
        sigmoid<<<32,32>>>(arr,32);
        arr=this->fc1.forward(arr,handle);
        sigmoid<<<16,16>>>(arr,16);
        arr=this->fc1.forward(arr,handle);
        sigmoid<<<4,4>>>(arr,4);
        cublasDestroy(handle);
        return arr;
    };
};

int main()
{
    float* array = (float*)malloc(32*32*sizeof(float));
    for(int i=0; i<32*32; i++)
        array[i]=rand()%255;
    float* cuarr;
    cudaMalloc(&cuarr,32*32);
    cudaMemcpy(cuarr,array,32*32,cudaMemcpyHostToDevice);
    printf("cuda first\n");
    Net net;
    printf("init net\n");
    cuarr = net.forward(cuarr);
    free(array);
    array=(float*)malloc(sizeof(float));
    cudaMemcpy(array,cuarr,1,cudaMemcpyDeviceToHost);
    cudaFree(cuarr);
    printf("%f\n",array);
    free(array);
    return 0;
}
