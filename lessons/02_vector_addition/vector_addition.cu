#include <stdio.h>

// CPU Version

// int main()

// {
//     const int N = 5;

//     int a[N] = {1, 2, 3, 4, 5};
//     int b[N] = {5, 4, 3, 2, 1};
//     int c[N] = {0, 0, 0, 0, 0};

//     for ( int i = 0; i < N; i++) {
//         c[i] = a[i] + b[i];
// }

// for ( int i = 0; i < N; i++) {
//     printf("%d \n", c[i]);}

//     return 0;}



// GPU Version

__global__ void addVectors(int *a, int *b, int *c, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 5;
    int a[N] = {1, 2, 3, 4, 5};
    int b[N] = {5, 4, 3, 2, 1};
    int c[N] = {0, 0, 0, 0, 0};
    int *d_a = nullptr;
    int *d_b = nullptr;
    int *d_c = nullptr;

    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMalloc((void**)&d_b, N*sizeof(int));
    cudaMalloc((void**)&d_c, N*sizeof(int));
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    addVectors<<<1, N>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    for ( int i = 0; i < N; i++) {
        printf("%d \n", c[i]);}
    return 0;}