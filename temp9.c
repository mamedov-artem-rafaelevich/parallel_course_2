#include <stdio.h>
#include <math.h>
#include <openacc.h>
#include <stdlib.h>
#include <nvToolsExt.h>
#define IDX2F(i,j,ld) (((j)-1)*(ld))+((i)-1)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

float max(float a, float b)
{
	if(a>b)
		return a;
	else 
		return b;
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
			setka[s*(s-1)+i]+=setka[s*(s-1)+i-1]+l1;
		}
		int iter=0;
		float err=1;
		if(s<16)
		{
			for(int i=0; i<s; i++)
			{
				for(int j=0; j<s; j++)
					printf("%f ",setka[i+s*j]);
				printf("\n");
			
			}
		}
//�������� ���� ���������, ����������� ������ � CPU �� GPU
#pragma acc data copyin(setka[0:s*s]) create(arr[0:s*s]) copy(s,iter,err,a,n)
		while(err>a && iter<n)
		{
//���������� ������, ������������� ���������� ��������. �������� ���������� �� GPU  � ����������� �� CPU
//#pragma acc kernels
			{
			iter++;
//#pragma acc update device(iter)
			err=0;
#pragma acc update device(err,iter)
			}
//#pragma acc loop
			//  gang vector vector_length()
//#pragma acc kernels
//����������� �������
#pragma acc loop
			{
			for(int i=0; i<s*s; i++)
			{
				arr[i]=setka[i];
//#pragma acc atomic update
//#pragma acc update device(arr[i])
//#pragma acc update host(arr[i])
#pragma acc update self(arr[i])
			}
			}
//#pragma acc kernels
//#pragma acc parallel loop gang num_gangs(4) vector vector_length(32)
//#pragma acc parallel loop reduction(max:err)
// loop gang vector(s*s)
#pragma acc data present(arr,setka)
//#pragma acc loop gang worker vector
//#pragma acc loop independent reduction(+:arr)
//#pragma acc loop gang vector collapse(2)
//#pragma acc parallel
{
//#pragma acc loop independent reduction(max:err)
//#pragma acc parallel loop reduction(max:err)
//#pragma acc parallel loop vector_length(32)/
#pragma acc parallel loop vector_length(32)
			for(int i=1; i<s-1; i++)
			{
//#pragma acc loop
//#pragma acc parallel loop
//#pragma acc loop vector
//�������� ������ �������� max
#pragma acc loop vector reduction(max:err)
				for(int j=1; j<s-1; j++)
				{
//					setka[IDX2C(i,j,s)]+=0.25*arr[IDX2C(i,j-1,s)];
//					setka[IDX2C(i,j,s)]+=0.25*arr[IDX2C(i,j+1,s)];
//					setka[IDX2C(i,j,s)]+=0.25*arr[IDX2C(i-1,j,s)];
//					setka[IDX2C(i,j,s)]+=0.25*arr[IDX2C(i+1,j,s)];
//					setka[i+j*(s-1)]=0.25*(arr[i+1+j*(s-1)]+arr[i-1+j*(s-1)]+arr[i+(j-1)*(s-1)]+arr[i+(j+1)*(s-1)]);
					setka[IDX2C(i,j,s)]=0.25*(arr[IDX2C(i,j-1,s)]+arr[IDX2C(i,j+1,s)]+arr[IDX2C(i-1,j,s)]+arr[IDX2C(i+1,j,s)]);
#pragma acc update self(setka[i+j*(s-1)])
//#pragma acc update device(setka[i+j*s])
//#pragma acc wait
					err=(float)max(err,setka[IDX2C(i,j,s)]-arr[IDX2C(i,j,s)]);
//���������� ������
//#pragma acc update host(err)
				}
			}
}
			if(iter%100==0 || iter==1)
				printf("%d %f \n",iter, err);
		}
		printf("Count iterations: %d\nError: %.10f\n", iter,err);
//�������� ������ � GPU
#pragma acc exit data delete(arr[:s*s]) delete (setka[:s*s])
		if(s<20)
			for(int i=0; i<s; i++)
			{
				for(int j=0; j<s; j++)
					printf("%f ",setka[i+s*j]);
				printf("\n");
			}
		free(arr);
		free(setka);
	}
	return 0;
}

