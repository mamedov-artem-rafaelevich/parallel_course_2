//#define CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double arr[10000000];
float brr[10000000];
int main()
{
	double ans = 0;
	float ans1 = 0;
	int i=0;
// #pragma acc data create(arr[:10000000]) copy(i)
{
//	#pragma acc kernels
	for(double a=0; a<2*3.1415926535; a+=2*0.00000031415926535)
	{
		arr[i]=sin(a);
		i++;
	}
}
	i=0;
//#pragma acc data copy(arr[:10000000]) copy(ans)
{
//	#pragma acc kernels
	{
		for(int k=0; k<10000000; k++)
			ans=ans+arr[k];
	}
}
// #pragma acc data create(arr[:10000000]) copy(i)
{
//	#pragma acc kernels
	for(float a=0; a<2*3.14159265535; a+=2*0.000000314159265535)
	{
		brr[i]=sinf(a);
		i++;
	}
}
//#pragma acc data copy(brr[:10000000]) copy(ans1)
{
//	#pragma acc kernels
	{
		for(int k=0; k<10000000; k++)
			ans1=ans1+brr[k];
	}
}
//	for(i=0; i<10000000; i+=4)
//		printf("%lf %lf %lf %lf %f %f %f %f\n",arr[i],arr[i+1],arr[i+2],arr[i+3],brr[i],brr[i+1],brr[i+2],brr[i+3]);
	printf("double: %lf\nfloat: %f\n", ans, ans1);
	return 0;
}
