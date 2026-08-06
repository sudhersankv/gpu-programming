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

__global__ void addVectors(int *a, int *b, int *c, int n) // Kernel function
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // Index of the thread
    if (idx < n) c[idx] = a[idx] + b[idx]; // Add the vectors
}

int main()
{
    const int N = 5;
    int a[N] = {1, 2, 3, 4, 5}; // Host memory
    int b[N] = {5, 4, 3, 2, 1}; 
    int c[N] = {0, 0, 0, 0, 0}; 
    int *d_a = nullptr; // Pointer to the device memory
    int *d_b = nullptr; 
    int *d_c = nullptr; 

    cudaMalloc((void**)&d_a, N*sizeof(int)); // Allocate memory on the device
    cudaMalloc((void**)&d_b, N*sizeof(int)); 
    cudaMalloc((void**)&d_c, N*sizeof(int)); 
    
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice); // Copy data from host to device
    cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice); 

    addVectors<<<1, N>>>(d_a, d_b, d_c, N); // Launch the kernel

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost); // Copy data from device to host

    cudaFree(d_a); // Free memory on the device
    cudaFree(d_b); 
    cudaFree(d_c); 
    for ( int i = 0; i < N; i++) {
        printf("%d \n", c[i]);} // Print the result
    return 0;}
