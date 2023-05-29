#include <math.h>
#include <cuda.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/10.2/include/cublas_v2.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

class Linear
{
    private:
    float* array;
    float* cuarray;
    float* cuout;
    int w,h;
    public:
    Linear(){
        this->w=0;
        this->h=0;
        this->array=NULL;
        this->cuarray=NULL;
//        this->cuout=NULL;
    };
    Linear(int a, int b)
    {
        this->array = (float*)malloc(a*b*sizeof(float));
        this->h=a;
        this->w=b;
        cudaMalloc(&this->cuout,this->w*sizeof(float));
        for(int i=0; i<a; i++)
            for(int j=0; j<b; j++)
                this->array[IDX2C(i,j,this->h)]=float(rand()%255);
//        cudaMalloc(&cuarray,a*b*sizeof(float));
        cudaMemcpy(this->cuarray,this->array,a*b*sizeof(float),cudaMemcpyHostToDevice);
    };
    Linear(const Linear& ln)
    {
        cudaMemcpy(this->cuarray,ln.cuarray,ln.w*ln.h,cudaMemcpyDeviceToDevice);
//        cudaMemcpy(this->cuout,ln.cuout,ln.w*ln.h,cudaMemcpyDeviceToDevice);
        this->array=ln.array;
        this->h=ln.h;
        this->w=ln.w;
    };
    ~Linear()
    {
        cudaFree(this->cuarray);
//        cudaFree(this->cuout);
        free(array);
    };

    Linear& operator=(const Linear& ln)
    {
        cudaMemcpy(this->cuarray,ln.cuarray,ln.w*ln.h,cudaMemcpyDeviceToDevice);
//        cudaMemcpy(this->cuout,ln.cuout,ln.w*ln.h,cudaMemcpyDeviceToDevice);
        this->array=ln.array;
        this->h=ln.h;
        this->w=ln.w;
        return *this;
    };

    void forward(float* arr)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float* skal;
        cudaMalloc(&skal,this->h*this->w*sizeof(float));
        float* cuout;
        cudaMalloc(&cuout,this->w*sizeof(float));
        float dop[this->w*this->h];
        for(int i=0; i<this->h*this->w; i++)
            dop[i]=1;
        cudaMemcpy(skal,dop,this->w*this->h*sizeof(float),cudaMemcpyHostToDevice);
        printf("start fc\n");
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,this->h,this->w,1,skal,arr,this->h,this->cuarray,this->w,skal,cuout,this->w);
        printf("cublas\n");
        cudaFree(arr);
        cudaMalloc(&arr,this->w*sizeof(float));
        std::swap(arr,cuout);
        cublasDestroy(handle);
        cudaFree(cuout);
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

    void forward(float* arr)
    {
        float arr1[32*32];
        cudaMemcpy(arr1,arr,32*32,cudaMemcpyDeviceToHost);
        for(int i=0; i<32; i++)
            printf("%f\n",arr1[i]);
        this->fc1.forward(arr);
        printf("first fc\n");
        sigmoid<<<16,16>>>(arr,16);
        for(int i=0; i<16; i++)
            printf("%f\n",arr[i]);
        this->fc1.forward(arr);
        sigmoid<<<4,4>>>(arr,4);
        for(int i=0; i<4; i++)
            printf("%f\n",arr[i]);
        this->fc1.forward(arr);
        sigmoid<<<1,1>>>(arr,1);
        printf("%f\n",arr[0]);
    };
};

int main()
{
    float* array = (float*)malloc(32*32*sizeof(float));
    for(int i=0; i<32*32; i++)
        array[i]=float(rand()%255)/255;
    float* cuarr;
    cudaMalloc(&cuarr,32*32);
    cudaMemcpy(cuarr,array,32*32,cudaMemcpyHostToDevice);
    printf("cuda first\n");
    Net net;
    printf("init net\n");
    net.forward(cuarr);
    free(array);
    array=(float*)malloc(sizeof(float));
    cudaMemcpy(array,cuarr,1,cudaMemcpyDeviceToHost);
//    std::cout << array[0];
    cudaFree(cuarr);
    printf("%f\n",array[0]);
    free(array);
    return 0;
}
