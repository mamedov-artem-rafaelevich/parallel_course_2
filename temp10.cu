#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/11.0/targets/x86_64-linux/include/cublas_v2.h"
#define IDX2F(i,j,ld) (((j)-1)*(ld))+((i)-1)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

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
		float* setka = (float*)calloc(s*s,sizeof(float));
		float* arr = (float*)calloc(s*s,sizeof(float));
		setka[0]=10;
		setka[s-1]=20;
		setka[(s-1)*s]=30;
		setka[s*s-1]=20;
		float l1=(10);
		l1/=s;
		float l2=20;
		l2/=s;
		for(int i=1; i<s-1; i++)
		{
			setka[i]=setka[i-1]+l1;
			setka[i*s]+=setka[(i-1)*s]+l2;
			setka[s-1+i*s]+=setka[s-1+(i-1)*s]+l1;
			setka[s*(s-1)+i]+=setka[s*(s-1)+i]+l1;
		}
		int iter=0;
		float err=1;
//		cublasInit();
		status = cublasCreate(&handle);
#pragma acc data copyin(setka[0:s*s]) create(arr[0:s*s]) copy(s,iter,err,a,n)
		while(err>a && iter<n)
		{
#pragma acc kernels
			{
			iter++;
			err=0;
			}
#pragma acc update host(err)
#pragma acc update host(iter)
#pragma acc parallel loop gang vector vector_length()
			for(int i=0; i<s*s; i++)
			{
				arr[i]=setka[i];
#pragma acc update device(setka[i])
			}
#pragma acc kernels
{
			for(int i=1; i<s-1; i++)
			{
				for(int j=1; j<s-1; j++)
				{
					setka[i+j*(s-1)]=0.25*(arr[i+1+j*(s-1)]+arr[i-1+j*(s-1)]+arr[i+(j-1)*(s-1)]+arr[i+(j+1)*(s-1)]);
#pragma acc update host(setka[i+j*(s-1)])
				}
			}
}
			int nm=0;
			float alpha=-1;
			status=cublasSaxpy(handle,s*s,&alpha,setka,1,arr,1);
			status=cublasIsamax(handle,s*s,arr,1,&nm);
			err=arr[nm];
			#pragma acc parallel
			if(iter%100==0 || iter==1)
				printf("%d %f \n",iter, err);
		}
		cublasDestroy(handle);
//		cublasShutdown();
		printf("Count iterations: %d", iter);
#pragma acc exit data delete(arr[:s*s]) delete (setka[:s*s])
		free(setka);
	}
	return 0;
}
