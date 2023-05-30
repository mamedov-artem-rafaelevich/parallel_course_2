#include <cub/cub.cuh>
#include <iostream>
#include <stdlib.h>

int main(){


  // Declare, allocate, and initialize device-accessible pointers for input and output
  int                      num_items = 32;
  double                      *d_in;
  double   *d_out;

  double *h_in = (double*)malloc(32*sizeof(double));//new double[num_items];
  double *h_out = (double*)malloc(sizeof(double));
  cudaMalloc(&d_in, num_items*sizeof(d_in[0]));
  cudaMalloc(&d_out, sizeof(double));
  for (int i = 0; i < num_items; i++) h_in[i] = (double)i;
  h_in[12] = 8;  // so we expect our return tuple to be 12,2
  cudaMemcpy(d_in, h_in, num_items*sizeof(d_in[0]), cudaMemcpyHostToDevice);

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run argmin-reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

  cudaMemcpy(h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
  printf("%f\n",h_out[0]);
//  std::cout << "maximum value: " << h_out[0] << std::endl;
  return 0;
}