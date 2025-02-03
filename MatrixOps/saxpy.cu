#include <cuda_runtime.h>
#include <iostream>
#define N 1024
#define THREADS_PER_BLOCK 16


__global__ void saxpy_kernel(float *x, float *y, float a ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) {
    y[idx] =  x[idx]*a + y[idx];
  }
}

int main() {
  float *x, *y, *d_x, *d_y;

  x = (float*)malloc( sizeof(float) * N);
  y = (float*)malloc( sizeof(float) * N);

  for(int i=0;i<N;i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMalloc((void**)&d_x, sizeof(float) * N);
  cudaMalloc((void**)&d_y, sizeof(float) * N);

  cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  saxpy_kernel <<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_x, d_y, 2.0f);


  cudaMemcpy(y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost);

  int flag = 0;

  for(int i=0;i<N;i++){
    if(abs(y[i] - (2.0 * x[i] + 2.0)) > 1e-6){
      flag = 1;
      break;
    }
  }

  if(flag){
    printf("Saxpy execution error\n");
  }
  else{
    printf("Saxpy execution success\n");
  }


}