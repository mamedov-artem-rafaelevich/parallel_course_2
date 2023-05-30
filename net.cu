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
    double* array;
    double* cuarray;
    double* cuout;
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
        this->array = (double*)malloc(a*b*sizeof(double));
        this->h=a;
        this->w=b;
        cudaMalloc(&this->cuout,this->w*sizeof(double));
        for(int i=0; i<a; i++)
            for(int j=0; j<b; j++)
                this->array[IDX2C(i,j,this->h)]=double(rand()%255)/255;
//        cudaMalloc(&cuarray,a*b*sizeof(double));
        cudaMemcpy(this->cuarray,this->array,a*b*sizeof(double),cudaMemcpyHostToDevice);
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

    void forward(double* arr)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        double* cuout;
        cudaMalloc(&cuout,this->w*sizeof(double));
//		double dop2[this->w];
//		cudaMemcpy(dop2,cuout,this->w*sizeof(double),cudaMemcpyDeviceToHost);
//		for(int i=0; i<this->h; i++)
//			printf("%d\n",this->cuarray[i]);
		double skal=1;
        printf("start fc\n");
        cublasDgemv(handle,CUBLAS_OP_T,this->w,this->h,&skal,arr,1,this->cuarray,1,&skal,cuout,1);
        printf("cublas\n");
        cudaFree(arr);
        cudaMalloc(&arr,this->w*sizeof(double));
        std::swap(arr,cuout);
        cublasDestroy(handle);
        cudaFree(cuout);
    };
};

__device__ double sigma(double a)
{
    return (1/(1-exp(-a)));
}

__global__ void sigmoid(double* arr,int n)
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

    void forward(double* arr)
    {
        double arr1[32*32];
        cudaMemcpy(arr1,arr,32*32*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i=0; i<32; i++)
            printf("%f\n",arr1[i]);
        this->fc1.forward(arr);
        printf("first fc\n");
        sigmoid<<<16,16>>>(arr,16);
		double arr2[16*16];
		cudaMemcpy(arr2,arr,16*16*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i=0; i<16; i++)
            printf("%f\n",arr2[i]);
        this->fc1.forward(arr);
        sigmoid<<<4,4>>>(arr,4);
		double arr3[4*4];
		cudaMemcpy(arr3,arr,4*4*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i=0; i<4; i++)
            printf("%f\n",arr3[i]);
        this->fc1.forward(arr);
        sigmoid<<<1,1>>>(arr,1);
		double arr0;
		cudaMemcpy(&arr0,arr,sizeof(double),cudaMemcpyDeviceToHost);
        printf("%f\n",arr0);
		printf("end forward net\n");
    };
};

int main()
{
    double* array = (double*)malloc(32*32*sizeof(double));
    for(int i=0; i<32*32; i++)
	{
        array[i]=(double(rand()%255))/255;
		std::cout << array[i] << '\n';
	}
    double* cuarr;
    cudaMalloc(&cuarr,32*32*sizeof(double));
    cudaMemcpy(cuarr,array,32*32,cudaMemcpyHostToDevice);
    printf("cuda first\n");
    Net net;
    printf("init net\n");
    net.forward(cuarr);
    free(array);
    array=(double*)malloc(sizeof(double));
    cudaMemcpy(array,cuarr,sizeof(double),cudaMemcpyDeviceToHost);
//    std::cout << array[0] << '\n';
    cudaFree(cuarr);
    printf("%f\n",array);
//    free(array);
    return 0;
}
