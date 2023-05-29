#include <stdio.h>
#include <math.h>
#include <openacc.h>
#include <stdlib.h>
#include <cuda.h>
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
	float* arr2 = (float*)calloc(s*s,sizeof(float));
    float* setka2 = (float*)calloc(s*s,sizeof(float));

    setka[0]=10;
    setka[s-1]=20;
    setka[(s-1)*s]=20;
    setka[s*s-1]=30;
	  arr[0]=10;
    arr[s-1]=20;
    arr[(s-1)*s]=20;
    arr[s*s-1]=30;
	  arr2[0]=10;
    arr2[s-1]=20;
    arr2[(s-1)*s]=20;
    arr2[s*s-1]=30;
    float l1=(10);
    l1/=s;
    float l2=20;
    l2/=s;
    int iter=0;
    float err=1;
    for(int i=1; i<s-1; i++)
    {
      setka[i]=setka[i-1]+l1;
      setka[i*s]+=setka[(i-1)*s]+l2;
      setka[s-1+i*s]+=setka[s-1+(i-1)*s]+l1;
      setka[s*(s-1)+i]+=setka[s*(s-1)+i-1]+l1;
      arr[i]=setka[i];
      arr[i*s]=setka[i*s];
      arr[s-1+i*s]=setka[s-1+i*s];
      arr[s*(s-1)+i]=setka[s*(s-1)+i];
      arr2[i]=setka[i];
      arr2[i*s]=setka[i*s];
      arr2[s-1+i*s]=setka[s-1+i*s];
      arr2[s*(s-1)+i]=setka[s*(s-1)+i];
    }

    if(s<16)
    {
      for(int i=0; i<s; i++)
      {
        for(int j=0; j<s; j++)
          printf("%f ",setka[i+s*j]);
        printf("\n");
      
      }
    }

    cublasStatus_t status;
    cublasHandle_t handle;

    #pragma acc data copyin(setka[0:s*s],arr[0:s*s],arr2[0:s*s],setka2[0:s*s],err,iter)
	{
    status = cublasCreate(&handle);
	  if (status != CUBLAS_STATUS_SUCCESS)
	  	printf("ERROR: %d!\n",status);
    while(err>a && iter<n)
    {
      iter++;

      #pragma acc kernels
      {
        err=0;
      }
#pragma acc data present(arr, setka)
#pragma acc parallel loop gang num_workers(4) vector_length(128) async(1)
      for(int i=1; i<s-1; i++)
      {
        #pragma acc loop vector 
        for(int j=1; j<s-1; j++)
        {
          arr[IDX2C(i,j,s)]=0.25*(setka[IDX2C(i,j-1,s)]+setka[IDX2C(i,j+1,s)]+setka[IDX2C(i-1,j,s)]+setka[IDX2C(i+1,j,s)]);
        }
      }
      float* dop;
      dop = arr;
      arr=setka;
      setka = dop;

//	  #pragma acc update device(setka,arr) async(1)
      if(iter%100==0 || iter==1)
      {
        int nm=0;
        float alpha[1];
        alpha[0]=-1;
        #pragma acc host_data use_device(arr,setka,arr2)
        {
    //	  #pragma acc update host(setka,arr) async(1)
  //      status=cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
          status=cublasScopy(handle,s*s,arr,1,arr2,1);
          if (status != CUBLAS_STATUS_SUCCESS)
            printf("ERROR: %d!\n",status);
          status=cublasSaxpy(handle,s*s,alpha,setka,1,arr2,1);
          if (status != CUBLAS_STATUS_SUCCESS)
            printf("ERROR: %d!\n",status);
          status=cublasIsamax(handle,s*s,arr2,1,&nm);
          if (status != CUBLAS_STATUS_SUCCESS)
            printf("ERROR: %d!\n",status);
          #pragma acc kernels
          {
          err = fabs(arr[nm-1]);
          }
          #pragma acc update host(err) async(1)
          #pragma acc wait(1)
          printf("%d %f\n",iter, err);
        }
        
//        #pragma acc wait(1) 
      }
    }
    status=cublasDestroy(handle);
    }
    printf("Count iterations: %d\nError: %.10f\n", iter,err);
	  if (status != CUBLAS_STATUS_SUCCESS)
	  	printf("ERROR: %d!\n",status);
    if(s<16)
      for(int i=0; i<s; i++)
      {
        for(int j=0; j<s; j++)
          printf("%f ",setka[i+s*j]);
        printf("\n");
      }
    free(arr);
    free(setka);
    free(setka2);
    free(arr2);
  }
  return 0;
}#include <stdio.h>
#include <math.h>
#include <openacc.h>
#include <stdlib.h>
#include <cuda.h>
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

    float* setka = (float*)calloc(s*s+1,sizeof(float));
    float* arr = (float*)calloc(s*s+1,sizeof(float));
	float* arr2 = (float*)calloc(s*s+1,sizeof(float));
    float* setka2 = (float*)calloc(s*s+1,sizeof(float));
	setka[0]=10;
	setka[s-1]=20;
	setka[(s-1)*s]=20;
	setka[s*s-1]=30;
	arr[0]=10;
	arr[s-1]=20;
	arr[(s-1)*s]=20;
	arr[s*s-1]=30;
	arr2[0]=10;
	arr2[s-1]=20;
	arr2[(s-1)*s]=20;
	arr2[s*s-1]=30;
    float l1=(10);
    l1/=s;
    float l2=20;
    l2/=s;
    int iter=0;
    float err=1;
    for(int i=1; i<s-1; i++)
    {
	    setka[i]=setka[i-1]+l1;
	setka[i*s]+=setka[(i-1)*s]+l2;
	setka[s-1+i*s]+=setka[s-1+(i-1)*s]+l1;
	setka[s*(s-1)+i]+=setka[s*(s-1)+i-1]+l1;
	arr[i]=setka[i];
	arr[i*s]=setka[i*s];
	arr[s-1+i*s]=setka[s-1+i*s];
	arr[s*(s-1)+i]=setka[s*(s-1)+i];
//	arr2[i]=setka[i];
//	arr2[i*s]=setka[i*s];
//	arr2[s-1+i*s]=setka[s-1+i*s];
//	arr2[s*(s-1)+i]=setka[s*(s-1)+i];
    }

    if(s<16)
    {
      for(int i=0; i<s; i++)
      {
        for(int j=0; j<s; j++)
          printf("%f ",setka[IDX2C(i,j,s)]);
        printf("\n");
      
      }
    }

    cublasStatus_t status;
    cublasHandle_t handle;
    printf("%d\n",CUBLAS_STATUS_MAPPING_ERROR);

    #pragma acc data copyin(setka[0:s*s],arr[0:s*s],arr2[0:s*s],setka2[0:s*s],err,iter)
	{
    status = cublasCreate(&handle);
	  if (status != CUBLAS_STATUS_SUCCESS)
	  	printf("ERROR: %d!\n",status);
    while(err>a && iter<n)
    {
      iter++;
	if(iter%100==0 || iter==1)
	{
      #pragma acc kernels
      {
        err=0;
      }
	}
#pragma acc data present(arr, setka)
#pragma acc parallel loop gang num_workers(4) vector_length(128) async(1)
      for(int i=1; i<s-1; i++)
      {
        #pragma acc loop vector 
        for(int j=1; j<s-1; j++)
        {
          arr[IDX2C(i,j,s)]=0.25*(setka[IDX2C(i,j-1,s)]+setka[IDX2C(i,j+1,s)]+setka[IDX2C(i-1,j,s)]+setka[IDX2C(i+1,j,s)]);
        }
      }
      float* dop;
      dop = arr;
      arr=setka;
      setka = dop;
	  #pragma acc update device(setka,arr) async(1)
      if(iter%100==0 || iter==1)
      {
        int nm=0;
        float alpha=-1;
        #pragma acc host_data use_device(arr,setka,arr2)
        {
    //	  #pragma acc update host(setka,arr) async(1)
  //      status=cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
          status=cublasScopy(handle,s*s,arr,1,arr2,1);
          if (status != CUBLAS_STATUS_SUCCESS)
            printf("ERROR 1: %d!\n",status);
#pragma acc kernels
	  {
	  if(s<16)
	      for(int i=0; i<s; i++)
	      {
		for(int j=0; j<s; j++)
		  printf("%f ",arr2[IDX2C(i,j,s)]);
		printf("\n");
	      }
	  }
          status=cublasSaxpy(handle,s*s,&alpha,setka,1,arr2,1);
          if (status != CUBLAS_STATUS_SUCCESS)
            printf("ERROR 2: %d!\n",status);
          status=cublasIsamin(handle,s*s,arr2,1,&nm);
          if (status != CUBLAS_STATUS_SUCCESS)
            printf("ERROR 3: %d!\n",status);
          #pragma acc kernels
          {
          err = fabs(arr[nm-1]);
          }
          #pragma acc update host(err) async(1)
          #pragma acc wait(1)
          printf("%d %f\n",iter, err);
        }
        
//        #pragma acc wait(1) 
      }
    }
    status=cublasDestroy(handle);
    }
    printf("Count iterations: %d\nError: %.10f\n", iter,err);
	  if (status != CUBLAS_STATUS_SUCCESS)
	  	printf("ERROR: %d!\n",status);
    if(s<16)
      for(int i=0; i<s; i++)
      {
        for(int j=0; j<s; j++)
          printf("%f ",setka[IDX2C(i,j,s)]);
        printf("\n");
      }
    free(arr);
    free(setka);
    free(setka2);
    free(arr2);
  }
  return 0;
}
