#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#define IDX2F(i,j,ld) (((j)-1)*(ld))+((i)-1)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void change(float* setka, float* arr, int s)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > s && i%s != 0 && i < s*(s - 1)-1 && i%s != s - 1)
		setka[i] = 0.25 * (arr[i-1] + arr[i+1] + arr[i+s] + arr[i-s]);
//	setka[IDX2C(i+threadIdx.x,j+threadIdx.y,s)]=0.25*(arr[IDX2C(i+threadIdx.x,j-1+threadIdx.y,s)]+arr[IDX2C(i+threadIdx.x,j+1+threadIdx.y,s)]+arr[IDX2C(i-1+threadIdx.x,j+threadIdx.y,s)]+arr[IDX2C(i+1+threadIdx.x,j+threadIdx.y,s)]);
}

__global__ void subtract_modulo_kernel(float* d_in1, float* d_in2, float* d_out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size*size) {
        float diff = d_in1[idx] - d_in2[idx];
        if(diff<0)
          d_out[idx]=-diff;
        else
          d_out[idx]=diff;
    }
}

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
  //  cudaSetDevice(3);
    float *cusetka;
    float *cuarr;
    float* cuarr2;
    cudaError_t stat;
    cudaStream_t stream;
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    float* max_value;
    stat=cudaMalloc((void**)&cusetka, s*s*sizeof(float));
    if(stat!=cudaSuccess)printf("err 1: %d", stat);
    stat=cudaMalloc((void**)&cuarr2, s*s*sizeof(float));
    if(stat!=cudaSuccess)printf("err 2: %d", stat);
    stat=cudaMalloc((void**)&cuarr, s*s*sizeof(float));
    if(stat!=cudaSuccess)printf("err 2: %d", stat);
    stat=cudaMemcpy(cuarr2, arr2, s*s*sizeof(float), cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)printf("err 3: %d", stat);
    stat=cudaMemcpy(cusetka, setka, s*s*sizeof(float), cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)printf("err 4: %d", stat);
    stat=cudaMemcpy(cuarr, arr, s*s*sizeof(float), cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)printf("err 5: %d", stat);
    stat=cudaMalloc((void**)&max_value, sizeof(float));
    if(stat!=cudaSuccess)printf("err 6: %d", stat);
    stat=cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, cuarr2, max_value, s*s);
    if(stat!=cudaSuccess)printf("err 7: %d", stat);
    stat=cudaMalloc(&d_temp_storage,temp_storage_bytes);
    if(stat!=cudaSuccess)printf("err 8: %d", stat);
    float* max_value_h=(float*)malloc(sizeof(float));
    while(err>a && iter<n)
    {
      iter++;
      if(iter%100==1)
        err=0;
        //Этого должно хватить для вычисления массива.
      cudaGraph_t graph;
      cudaGraphExec_t instance;
//      cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
      change<<<s, s, 0 >>>(cusetka, cuarr, s);
//      cudaStreamEndCapture(stream, &graph);
//      cudaGraphInstantiate(&instance,graph,NULL,NULL,0);
//      change<<<blocksPerGrid, threadsPerBlock >>>(cuarr, cusetka, n);
      if(iter%100==1)
      {
        subtract_modulo_kernel<<<s, s, 0>>>(cusetka, cuarr, cuarr2, s);
        // cudaMemcpy(setka,cuarr2,s*s*sizeof(float),cudaMemcpyDeviceToHost);
        // if(s<16)
        // {
        //   for(int i=0; i<s; i++)
        //   {
        //     for(int j=0; j<s; j++)
        //       printf("%f ",setka[i+s*j]);
        //     printf("\n");
          
        //   }
        // }
        // const int block_size = 256;
        // const int num_blocks = (n + block_size - 1) / block_size;

//        cudaDeviceSynchronize();

        stat=cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, cuarr2, max_value, s*s);
        if(stat!=cudaSuccess)printf("%d\n",stat);
        cudaMemcpy(max_value_h,max_value,sizeof(float),cudaMemcpyDeviceToHost);
        err=max_value_h[0];
        printf("%d %f\n", iter, err);
      }

      float* dop;
      dop = cuarr;
      cuarr=cusetka;
      cusetka = dop;
      // cudaMemcpy(cuarr2,cuarr,s*s*sizeof(float),cudaMemcpyDeviceToDevice);
      // cudaMemcpy(cuarr,cusetka,s*s*sizeof(float),cudaMemcpyDeviceToDevice);
      // cudaMemcpy(cusetka,cuarr2,s*s*sizeof(float),cudaMemcpyDeviceToDevice);
      //std::swap(cuarr,cusetka);
    }
    cudaMemcpy(setka,cusetka,s*s*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(arr, cuarr, s*s*sizeof(float), cudaMemcpyDeviceToHost);
    free(max_value_h);
    cudaFree(d_temp_storage);
    cudaFree(cusetka);
    cudaFree(cuarr);
    printf("Count iterations: %d\nError: %.10f\n", iter,err);
    if(s<16)
    {
      for(int i=0; i<s; i++)
      {
        for(int j=0; j<s; j++)
          printf("%f ",setka[i+s*j]);
        printf("\n");
      }
    }
    free(setka);
    free(arr);
  }
  return 0;
}
