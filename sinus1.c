#define CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void function(double* arr)
{
	int i=0;
	for(double a=0; a<2*3.1415192; a+=2*0.0000003141519)
	{
		arr[i]=sin(a);
		i++;
	}
}
void function1(float* arr)
{
	int i=0;
	for(float a=0; a<2*3.1415192; a+=2*0.0000003141519)
	{
		arr[i]=sinf(a);
		i++;
	}
}
double summator(double* arr)
{
	double ans=0;
#pragma acc kernels
	{
		for(int k=0; k<10000000; k++)
			ans+=arr[k];
	}
	return ans;
}
float summator1(float* arr)
{
	float ans=0;
#pragma acc kernels
	{
		for(int k=0; k<10000000; k++)
			ans+=arr[k];
	}
	return ans;
}

int main()
{
	double* arr=(double*)malloc(80000000);
	float* brr=(float*)malloc(40000000);
	function(arr);
	function1(brr);
	double d = summator(arr);
	double d1 = summator1(arr);
	printf("double: %lf\nfloat: %f\n", d, d1);
//	for(int i=0; i<10000000; i+=4)
//		printf("%lf %lf %lf %lf\n", arr[i],arr[i+1],arr[i+2],arr[i+3]);
	return 0;
}
