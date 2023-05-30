#include <stdio.h>                                                                                                   
#include <cuda.h>                                                                                                   
 #include <cuda_runtime.h>
 __global__ void pri()   
{                                                                                                                            
printf("Hello\n");                                                                                               
}                                                                                                                   
 int main()                                                                                                           
{                                                                                                                            
pri<<<1,1>>>();                                                                                                      
cudaDeviceSynchronize();                                                                                             
printf("%s\n", cudaGetErrorString(cudaGetLastError()));                                                      
}