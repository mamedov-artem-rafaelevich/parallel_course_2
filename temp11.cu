#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/11.0/targets/x86_64-linux/include/cublas_v2.h"
#define IDX2F(i,j,ld) (((j)-1)*(ld))+((i)-1)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__device__ void change(float* setka, float* arr, int i, int j, int s)
{
	setka[IDX2C(i+threadIdx.x,j+threadIdx.y,s)]=0.25*(arr[IDX2C(i+threadIdx.x,j-1+threadIdx.y,s)]+arr[IDX2C(i+threadIdx.x,j+1+threadIdx.y,s)]+arr[IDX2C(i-1+threadIdx.x,j+threadIdx.y,s)]+arr[IDX2C(i+1+threadIdx.x,j+threadIdx.y,s)]);
}

/*_device__ void init(int s, int i, float l1, float l2)
{
	setka[i]=setka[i-1]+l1;
	setka[i*s]+=setka[(i-1)*s]+l2;
	setka[s-1+i*s]+=setka[s-1+(i-1)*s]+l1;
	setka[s*(s-1)+i]+=setka[s*(s-1)+i]+l1;
}*/

__device__ void deliter(float* setka, float* arr, float* err, int i)
{
	err[i+threadIdx.x]=setka[i+threadIdx.x]-arr[i+threadIdx.x];
}

int main(int argc, char** argv)
{
	float a=0;
	int s=0;
	int n=0;
	cublasStatus_t status;
	cublasHandle_t handle;
	if(argv[1][1]=='h')
	{
		printf("Put -h to show this.\n");
		printf("Put -a <NUMBER_OF_ACCURACY*10^6> -s <SIZE^2> -n <NUMBER_OF_ITERATION*10^6>.\n");
	}
	else
	{
		for(int k=1; k<argc; k+=2)
		{
			if(argv[k][1]=='a')
				a=(float)atof(argv[k+1]);
			else if(argv[k][1]=='s')
				s=atoi(argv[k+1]);
			else if(argv[k][1]=='n')
				n=atoi(argv[k+1]);
		}
		float* setka;
		cudaMalloc((void**)setka,s*s*sizeof(float));
		float* arr;
		cudaMalloc((void**)arr,s*s*sizeof(float));
		float* errors;
		cudaMalloc((void**)errors,s*s*sizeof(float));
		setka[0]=10;
		setka[s-1]=20;
		setka[(s-1)*s]=30;
		setka[s*s-1]=20;
		float l1=(10);
		l1/=s;
		float l2=20;
		l2/=s;
		for(int i=1; i<s-1; i+=32)
		{
//			init<<<2,5>>>(s,i,l1,l2);
			setka[i]=setka[i-1]+l1;
			setka[i*s]+=setka[(i-1)*s]+l2;
			setka[s-1+i*s]+=setka[s-1+(i-1)*s]+l1;
			setka[s*(s-1)+i]+=setka[s*(s-1)+i]+l1;
		}
		cudaDeviceSynchronize();
		int iter=0;
		float err=1;
//		cublasInit();
		while(err>a && iter<n)
		{
			iter++;
			err=0;
			for(int i=0; i<s*s; i++)
			{
				arr[i]=setka[i];
			}
			for(int i=1; i<s-1; i+=32)
			{
				for(int j=1; j<s-1; j+=32)
				{
//					setka[i+j*(s-1)]=0.25*(arr[i+1+j*(s-1)]+arr[i-1+j*(s-1)]+arr[i+(j-1)*(s-1)]+arr[i+(j+1)*(s-1)]);
					change<<<2,32>>>(setka,arr,i,j,s);
				}
			}
			for(int i=0; i<s*s; i+=32
				deliter<<<2,32>>>(setka,arr,errors,i);
			for(int i=0; i<s*s; i++)
				if(err<errors[i])
					err=errors[i];
			//
			if(iter%100==0 || iter==1)
				printf("%d %f \n",iter, err);
		}
		cudaFree(arr);
		cudaFree(setka);
		cudaFree(errors);
		cublasDestroy(handle);
//		cublasShutdown();
		printf("Count iterations: %d", iter);
		free(setka);
	}
	return 0;
}
