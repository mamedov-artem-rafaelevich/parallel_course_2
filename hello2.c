#include "cuda_runtime.h"
#include <stdio.h>

__global__ void hello()
{
	printf("Hello from %d, %d",blockIdx.x,threadIdx.x);
}

int main()
{
	hello<<<2,5>>>();
	return 0;
}
