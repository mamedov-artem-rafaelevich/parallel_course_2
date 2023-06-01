#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main()
{
    const int n = 3, m = 4; // размер матрицы
    const float alpha = 1.0f, beta = 0.0f;
    float *h_A, *h_x, *h_y;
    float *d_A, *d_x, *d_y;
    cublasHandle_t handle;

    // выделение памяти на хосте
    h_A = (float*)malloc(n * n * sizeof(float));
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));

    // заполнение матрицы и векторов
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            h_A[i * n + j] = 0.5;//i + j;
        }
        h_x[i] = 0.5;//i;
//        h_y[i] = 1;//0;
    }

    // выделение памяти на устройстве
    cudaMalloc((void**)&d_A, n * m * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, m * sizeof(float));

    // копирование данных на устройство
    cudaMemcpy(d_A, h_A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);

    // создание объекта handle для работы с cuBLAS
    cublasCreate(&handle);

    // выполнение умножения матрицы на вектор
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);

    // копирование результата на хост
    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    // вывод результата
    for (int i = 0; i < n; i++) {
        printf("%f ", h_y[i]);
    }

    // освобождение памяти
    free(h_A);
    free(h_x);
    free(h_y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);

    return 0;
}