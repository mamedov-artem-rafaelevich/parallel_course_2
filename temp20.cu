#include <ctime>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#define IDX2F(i,j,ld) (((j)-1)*(ld))+((i)-1)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
//Функция для вычисления теплопроводности по пятиточечному шаблону
__global__ void change(double* setka, double* arr, int s)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > s && i%s != 0 && i < s*(s - 1)-1 && i%s != s - 1)
		setka[i] = 0.25 * (arr[i-1] + arr[i+1] + arr[i+s] + arr[i-s]);
}
//Функция для вычисления разницы между итерациями
__global__ void subtract_modulo_kernel(double* d_in1, double* d_in2, double* d_out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx <= size*size) {
        double diff = d_in1[idx] - d_in2[idx];
        if(diff<0)
          d_out[idx]=-diff;
        else
          d_out[idx]=diff;
    }
}

int main(int argc, char** argv)
{
  double a=0;
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
        a=(double)atof(argv[k+1]);
      else if(argv[k][1]=='s')
        s=atoi(argv[k+1]);
      else if(argv[k][1]=='n')
        n=atoi(argv[k+1]);
    }
//Инициализация
    double* setka = (double*)calloc(s*s,sizeof(double));
    double* arr = (double*)calloc(s*s,sizeof(double));
    double* arr2 = (double*)calloc(s*s,sizeof(double));

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
    double l1=(10);
    l1/=s-1;
    double l2=20;
    l2/=s-1;
    int iter=0;
    double err=1;
    for(int i=1; i<s-1; i++)
    {
      setka[i]=setka[i-1]+l1;
      setka[i*s]+=setka[(i-1)*s]+l1;
      setka[s-1+i*s]+=setka[s-1+(i-1)*s]+l1;
      setka[s*(s-1)+i]+=setka[s*(s-1)+i-1]+l1;
      arr[i]=setka[i];
      arr[i*s]=setka[i*s];
      arr[s-1+i*s]=setka[s-1+i*s];
      arr[s*(s-1)+i]=setka[s*(s-1)+i];
    }
//Визуализация сеток, меньших, чем 16
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
    double *cusetka;
    double *cuarr;
    double* cuarr2;
    cudaError_t stat;
    cudaStream_t stream;
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    double* max_value;
    //Выделение памяти на видеокарте
    stat=cudaMalloc((void**)&cusetka, s*s*sizeof(double));
    if(stat!=cudaSuccess)printf("err 1: %d", stat);
    stat=cudaMalloc((void**)&cuarr2, s*s*sizeof(double));
    if(stat!=cudaSuccess)printf("err 2: %d", stat);
    stat=cudaMalloc((void**)&cuarr, s*s*sizeof(double));
    if(stat!=cudaSuccess)printf("err 2: %d", stat);
    stat=cudaMemcpy(cuarr2, arr2, s*s*sizeof(double), cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)printf("err 3: %d", stat);
    stat=cudaMemcpy(cusetka, setka, s*s*sizeof(double), cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)printf("err 4: %d", stat);
    stat=cudaMemcpy(cuarr, arr, s*s*sizeof(double), cudaMemcpyHostToDevice);
    if(stat!=cudaSuccess)printf("err 5: %d", stat);
    stat=cudaMalloc((void**)&max_value, sizeof(double));
    if(stat!=cudaSuccess)printf("err 6: %d", stat);
    //Инициализация cub::DeviceReduce::Max
    stat=cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, cuarr2, max_value, s*s);
    if(stat!=cudaSuccess)printf("err 7: %d", stat);
    stat=cudaMalloc(&d_temp_storage,temp_storage_bytes);
    if(stat!=cudaSuccess)printf("err 8: %d", stat);
    double* max_value_h=(double*)malloc(sizeof(double));
    cudaGraph_t graph;
//    cudaGraphExec_t instance;
    //Основной цикл
	std::time_t result = std::time(nullptr);
    while(err>a && iter<n)
    {
      iter++;
      if(iter%100==1)
        err=0;
        //Этого должно хватить для вычисления массива.
//Вычисление слоя
//Количество потоеков в рамках потоковогоо блока должно быть не больше 1024 и кратно 32.
//Найти количество блоков в сетке , исходя из количества потоков в сетке; исправить заполнение границ; добавить cudaGraph; замерить время внутри кода (библиотеки time).
//Разобраться, почему выводится ноль в результате вычислений. Заменить double на double
      change<<<s, s, 0>>>(cusetka, cuarr, s);
      if(iter%100==1)
      {
        //Вычисление слоя с ошибкой
        subtract_modulo_kernel<<<s, s, 0>>>(cusetka, cuarr, cuarr2, s);
//Вычисление ошибки
        stat=cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, cuarr2, max_value, s*s);
        if(stat!=cudaSuccess)printf("%d\n",stat);
        cudaMemcpy(max_value_h,max_value,sizeof(double),cudaMemcpyDeviceToHost);
        err=max_value_h[0];
        printf("%d %.6f\n", iter, err);
      }
//Копирование
      double* dop;
      dop = cuarr;
      cuarr=cusetka;
      cusetka = dop;
    }
	result = std::time(nullptr) - result;
    //Возвращение данныз на хост
    cudaMemcpy(setka,cusetka,s*s*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(arr, cuarr, s*s*sizeof(double), cudaMemcpyDeviceToHost);
    free(max_value_h);
    cudaFree(d_temp_storage);
    cudaFree(cusetka);
    cudaFree(cuarr);
    printf("Count iterations: %d\nError: %.8f\nTime: %d\n", iter,err,result);
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
