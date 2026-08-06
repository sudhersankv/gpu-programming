#include <stdio.h>

// CPU Version

// int main()

// {
//     const int ROWS = 2;
//     const int COLS = 3;

//     int a[ROWS][COLS] = { {1,2,3}, {4,5,6}};

//     int b[ROWS][COLS] = { {7,8,9}, {10,11,12}};

//     int c[ROWS][COLS] = {0};

//     for (int i = 0; i < ROWS; i++) {
//         for (int j = 0; j < COLS; j++) {
//             c[i][j] = a[i][j] + b[i][j];}}

//             printf("Result: \n");
//             for (int i = 0; i < ROWS; i++) {
//                 for (int j = 0; j < COLS; j++) {
//                     printf("%d ", c[i][j]);
//                 }
//                 printf("\n");
//             }
//             return 0;

// }

// GPU Version


__global__ void matrixAddition(int* a, int*b, int*c, int rows, int cols)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x; // Row index
    int col = blockIdx.y*blockDim.y + threadIdx.y; // Column index
    if (row < rows && col < cols) {
        c[row*cols + col] = a[row*cols + col] + b[row*cols + col]; // Add the elements of the matrices
}

    
}
int main()
{
    const int ROWS = 2; // Number of rows
    const int COLS = 3; // Number of columns
    int a[ROWS][COLS] = { {1,2,3}, {4,5,6}}; // Matrix A
    int b[ROWS][COLS] = { {7,8,9}, {10,11,12}}; // Matrix B
    int c[ROWS][COLS] = {0}; // Matrix C
    int *d_a, *d_b, *d_c; // Pointers to the device memory
    cudaMalloc((void**)&d_a, ROWS*COLS*sizeof(int)); // Allocate memory on the device
    cudaMalloc((void**)&d_b, ROWS*COLS*sizeof(int)); 
    cudaMalloc((void**)&d_c, ROWS*COLS*sizeof(int)); 
    cudaMemcpy(d_a, a, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice); // Copy data from host to device
    cudaMemcpy(d_b, b, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice); 
    int c[ROWS][COLS] = {0};
    int *d_a, *d_b, *d_c; // Pointers to the device memory
    cudaMalloc((void**)&d_a, ROWS*COLS*sizeof(int)); // Allocate memory on the device
    cudaMalloc((void**)&d_b, ROWS*COLS*sizeof(int));
    cudaMalloc((void**)&d_c, ROWS*COLS*sizeof(int));
    cudaMemcpy(d_a, a, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
    dim3 threads(ROWS, COLS); // Number of threads in the block
    matrixAddition<<<1, threads>>>(d_a, d_b, d_c, ROWS, COLS); // Launch the kernel
    cudaMemcpy(c, d_c, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a); // Free memory on the device
    cudaFree(d_b);
    cudaFree(d_c);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }
    return 0;
}