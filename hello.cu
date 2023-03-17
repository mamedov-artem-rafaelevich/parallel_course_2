#include "cuda_runtime.h"
#include <stdio.h>

__global__ void hello(float* a,char* s)
{
	a[0]+=a[1];
	s[0]='h';
	s[1]='e';
	s[2]='l';
	s[3]='l';
	s[4]='o';
	printf("%s\n",s);
}

int main()
{
	float* a=(float*)malloc(8);
	a[0]=2;
	a[1]=3;
	char* s;
	cudaMalloc(&s,5);
	float *b;
	cudaMalloc(&b,8);
	cudaMemcpy(b,a,8,cudaMemcpyHostToDevice);
	hello<<<2,5>>>(b,s);
	cudaDeviceSynchronize();
	cudaMemcpy(a,b,8,cudaMemcpyDeviceToHost);
	cudaFree(b);
	cudaFree(s);
	printf("\n%f %f\n",a[0],a[1]);
	free(a);
	return 0;
}
