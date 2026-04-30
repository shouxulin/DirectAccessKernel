#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>




void print_device_name(int device_id) {
//   int device_id;
  cudaGetDevice(&device_id);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  printf("Running on device %d: %s\n", device_id, prop.name);
}






