#include <stdio.h>
#include <openacc.h>
#include <math.h>
#include <stdlib.h>

__global__ void function(float a, int s, int n)
{
	float* setka = (float*)cudaCalloc(s*s,sizeof(float));
	float* arr = (float*)cudaCalloc(s*s,sizeof(float));
	setka[0]=10;
	setka[s-1]=20;
	setka[(s-1)*s]=30;
	setka[s*s-1]=20;
	int iter=0;
	float err=0;
#pragma acc data copyin(setka[0:s*s]) create(arr[0:s*s]) copy(s,iter,err)
	while(err<a && iter<n)
	{
		iter++;
		err=0;
#pragma acc parallel loop
		for(int i=0; i<s*s; i++)
		{
#pragma acc atomic update
			arr[i]=setka[i];
		}
#pragma acc parallel loop
		for(int i=1; i<s-1; i++)
			for(int j=1; j<s-1; j++)
			{
				setka[i+j*(s-1)]=0.25*(arr[i+1+j*(s-1)]+arr[i-1+j*(s-1)]+arr[i+(j-1)*(s-1)]+arr[i+(j+1)*(s-1)]);
				if(err<arr[i+j*(s-1)])
					err=arr[i+j*(s-1)];
			}
		if(iter%100==0 || iter==1)
			printf("%d %f \n",iter, err);
	}
#pragma acc exit data delete(arr[:s*s]) delete (setka[:s*s])
	cudaFree(setka);
};

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
		dim3 BS(512);
		dim3 GS(10);
		function<<<GS, BS>>>(a,s,n);
	}	
	return 0;
}
