#include <stdio.h>
#include <math.h>
#include <openacc.h>
#include <stdlib.h>
#include <nvToolsExt.h>
#define IDX2F(i,j,ld) (((j)-1)*(ld))+((i)-1)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//Общий вид командной строки: temp17 -a 0.000001 -s 1024 -n 1000000, при этом порядок не имеет значения

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
//Парсинг командной строки
		for(int k=1; k<argc; k+=2)
		{
			if(argv[k][1]=='a')
				a=(float)atof(argv[k+1]);
			else if(argv[k][1]=='s')
				s=atoi(argv[k+1]);
			else if(argv[k][1]=='n')
				n=atoi(argv[k+1]);
		}
//Выделение памяти
		float* setka = (float*)calloc(s*s,sizeof(float));
		float* arr = (float*)calloc(s*s,sizeof(float));
//Инициализация
		setka[0]=10;
		setka[s-1]=20;
		setka[(s-1)*s]=20;
		setka[s*s-1]=30;
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
//Если размер меньше 16, распечатать массив
        if(s<16)
		{
			for(int i=0; i<s; i++)
			{
				for(int j=0; j<s; j++)
					printf("%f ",setka[i+s*j]);
				printf("\n");
			
			}
		}
//Копирование данных на устройство
        #pragma acc data copy(setka[0:s*s],arr[0:s*s]) copyin(err,iter)
{
		while(err>a && iter<n)
		{
			iter++;
//Выполнение на устройстве
				if(iter%100==0 || iter==1)
				{
				#pragma acc kernels
				{
				err=0;
				}
				}
//При заходе в область кода данные должны уже быть представлены в памяти устройства, а число структурированных ссылок должно быть инкрементировано. При покидании области кода число структурированных ссылок должно юыть декрементировано.
#pragma acc data present(arr, setka)
#pragma acc parallel loop independent vector vector_length(256) gang num_gangs(256) async(1)
			for(int i=1; i<s-1; i++)
			{
				if(iter%100==0 || iter==1)
				{
					#pragma acc loop vector reduction(max:err)
				}
				for(int j=1; j<s-1; j++)
				{
					arr[IDX2C(i,j,s)]=0.25*(setka[IDX2C(i,j-1,s)]+setka[IDX2C(i,j+1,s)]+setka[IDX2C(i-1,j,s)]+setka[IDX2C(i+1,j,s)]);
					err=fmax(err,fabs(arr[IDX2C(i,j,s)]-setka[IDX2C(i,j,s)]));
				}
			}
			for()
			float* dop;
			dop = arr;
			arr=setka;
			setka = dop;
			if(iter%100==0 || iter==1)
			{
//Обновление ошибки на хосте
			#pragma acc update host(err) async(1)
			#pragma acc wait(1) 
				printf("%d %f \n",iter, err);
			}
		}
		printf("Count iterations: %d\nError: %.10f\n", iter,err);
}
		if(s<16)
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

//