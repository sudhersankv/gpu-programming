#include <stdio.h>

__global__ void helloFromGPU() { printf("Hello from thread %d\n", threadIdx.x);}

int main() {

    helloFromGPU<<<2, 5>>>();

    cudaDeviceSynchronize();

    return 0;}