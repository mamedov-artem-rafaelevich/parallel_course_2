#include <math.h>
#include <cuda.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

class Linear
{
    private:
    float* array;
    float* cuarray;
    int w,h;
    public:
    Linear(){
        this->w=0;
        this->h=0;
        this->array=NULL;
        this->cuarray=NULL;
//        this->cuout=NULL;
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
//	cudaMalloc(&this->cuarray,ln.h*ln.w*sizeof(float));
//        cudaMemcpy(this->cuarray,ln.cuarray,ln.w*ln.h,cudaMemcpyDeviceToDevice);
//        cudaMemcpy(this->cuout,ln.cuout,ln.w*ln.h,cudaMemcpyDeviceToDevice);
	this->cuarray=ln.cuarray;
        this->array=ln.array;
        this->h=ln.h;
        this->w=ln.w;
        return *this;
    };
    void initLinear(int a, int b)
    {
        this->array = (float*)malloc(a*b*sizeof(float));
        this->h=a;
        this->w=b;
	FILE* fl;
	if(a==1024)fl = fopen("fc1.bin","rb");
	else if(a==256)fl = fopen("fc2.bin","rb");
	else fl = fopen("fc3.bin","rb");
	fread(this->array,sizeof(float),a*b,fl);
	fclose(fl);
//	for(int i=0; i<16; i++)
//		printf("%f\n",this->array[i]);
        cudaMalloc(&this->cuarray,a*b*sizeof(float));
//        for(int i=0; i<a; i++)
//            for(int j=0; j<b; j++)
//	    {
//                this->array[IDX2C(j,i,a*b)]=float(rand()%255)/255;
//		printf("%f ",this->array[IDX2C(j,i,a*b)]);
//	    }
        cudaMemcpy(this->cuarray,this->array,a*b*sizeof(float),cudaMemcpyHostToDevice);
    };

    void forward(float* arr)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float* cuout;
        cudaMalloc(&cuout,this->w*sizeof(float));
//	float dop2[this->w];
//	cudaMemcpy(dop2,this->cuarray,this->w*sizeof(float),cudaMemcpyDeviceToHost);
//	printf("weights\n");
//	for(int i=0; i<4; i++)
//		printf("%f\n",dop2[i]);
	float skal=1;
//        printf("start fc\n");
//	cublasStatus_t stat;
        cublasSgemv(handle,CUBLAS_OP_N,this->w,this->h,&skal,this->cuarray,this->w,arr,1,&skal,cuout,1);
//        printf("%d\n",stat);
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
    if(IDX2C(threadIdx.x,blockIdx.x,n)<n*n)
    	arr[IDX2C(threadIdx.x,blockIdx.x,n)]=sigma(arr[IDX2C(threadIdx.x,blockIdx.x,n)]);
}

class Net
{
    public:
    Linear fc1,fc2,fc3;
    
    Net()
    {
//        printf("init\n");
	this->fc1.initLinear(32*32,16*16);
	this->fc2.initLinear(16*16,4*4);
	this->fc3.initLinear(4*4,1);
//        this->fc1=Linear(32*32,16*16);
//        this->fc2=Linear(16*16,4*4);
//        this->fc3=Linear(4*4,1);
    };

    void forward(float* arr)
    {
        float arr1[32*32];
//        cudaMemcpy(arr1,arr,32*32*sizeof(float),cudaMemcpyDeviceToHost);
//	for(int i=0; i<32; i++)
//		std::cout << arr1[i] << "\n";
        this->fc1.forward(arr);
//       	cudaMemcpy(arr1,arr,32*32*sizeof(float),cudaMemcpyDeviceToHost);
//	for(int i=0; i<32; i++)
//		std::cout << arr1[i] << "\n";
//        printf("first fc---------------------------------------------------------\n");
        sigmoid<<<16,16>>>(arr,16);
//	float arr2[16*16];
//	cudaMemcpy(arr2,arr,16*16*sizeof(float),cudaMemcpyDeviceToHost);
//        for(int i=0; i<16; i++)
//            printf("%f\n",arr2[i]);
        this->fc2.forward(arr);
        sigmoid<<<4,4>>>(arr,4);
//	float arr3[4*4];
//	cudaMemcpy(arr3,arr,4*4*sizeof(float),cudaMemcpyDeviceToHost);
//        for(int i=0; i<4; i++)
//            printf("%f\n",arr3[i]);
        this->fc3.forward(arr);
        sigmoid<<<1,1>>>(arr,1);
	float arr0;
//	cudaMemcpy(&arr0,arr,sizeof(float),cudaMemcpyDeviceToHost);
//        printf("%f\n",arr0);
//	printf("end forward net\n");
    };
};

int main()
{
    float* array = (float*)malloc(32*32*sizeof(float));
    FILE* fl;
    fl = fopen("start.bin","rb");
    fread(array,sizeof(float),32*32,fl);
    fclose(fl);
//    for(int i=0; i<32*32; i++)
//    {
//        array[i]=(float(rand()%255))/255;
//	std::cout << array[i] << '\n';
//    }
    float* cuarr;
    cudaMalloc(&cuarr,32*32*sizeof(float));
    cudaMemcpy(cuarr,array,32*32,cudaMemcpyHostToDevice);
//    printf("cuda first\n");
    Net net;
//    printf("init net\n");
    net.forward(cuarr);
    free(array);
    array=(float*)malloc(sizeof(float));
    cudaMemcpy(array,cuarr,sizeof(float),cudaMemcpyDeviceToHost);
//    std::cout << array[0] << '\n';
    cudaFree(cuarr);
    printf("%f\n",array[0]);
    free(array);
    return 0;
}
