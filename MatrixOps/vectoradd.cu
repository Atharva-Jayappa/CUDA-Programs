#include <iostream>
#include <cuda_runtime.h>
#define N (1024*1024)
#define T 512



__global__ void add(const float* d_A,const float* d_B,float* d_C ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        d_C[idx] = d_A[idx] + d_B[idx];
    }

}

int main() {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    h_A = (float*)malloc(N*sizeof(float));
    h_B = (float*)malloc(N*sizeof(float));
    h_C = (float*)malloc(N*sizeof(float));

    cudaMalloc((void**)&d_A, N*sizeof(float));
    cudaMalloc((void**)&d_B, N*sizeof(float));
    cudaMalloc((void**)&d_C, N*sizeof(float));

    for (int i = 0; i < N; i++){
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }

    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);


    add<<<(N+T-1)/T, T>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);


    printf("h_A : %f", h_A[0]);
    printf("h_B : %f", h_B[0]);
    printf("h_C : %f", h_C[0]);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);




    return 0;
}
