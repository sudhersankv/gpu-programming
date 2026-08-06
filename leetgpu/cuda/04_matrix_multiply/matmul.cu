#include <stdio.h>

// CPU Implementation of Matrix Multiplication
// int main()
// {
//     const int M = 3;
//     const int K = 3;
//     const int N = 2;


//     int A[M][K] = {{2,4,6}, {1,4,5}, {5,3,6}};
//     int B[K][N] = {{2,3}, {4,5}, {6,7}};
//     int C[M][N] = {0};

//     for(int i = 0; i<M; i++)
//     {
//         for(int j = 0; j<N; j++)
//         {
//             for(int k = 0; k<K; k++)
//         {
//             C[i][j] += A[i][k] * B[k][j];

//         }}
//     }

//     for(int i = 0; i<M; i++)
//     {
//         for(int j = 0; j<N; j++)
//         {
//             printf("%d ", C[i][j]);
//         }
//         printf("\n");
//     }

// }

__global__ void matrixMultiply(int *A, int *B, int *C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index of the current thread
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index of the current thread

    if(row < M && col < N) // check if the current thread is within the bounds of the matrices
    {
        int sum = 0;
        
        for(int k = 0; k<K; k++)
        {
            sum += A[row *K + k] * B[k * N + col]; // multiply the elements of the matrices and add to the sum
        }
        C[row * N + col] = sum; // store the result in the corresponding element of the result matrix
    }
}

int main()
{
    const int M = 3;
    const int K = 3;
    const int N = 2;

    int A[M][K] = {{2,4,6}, {1,4,5}, {5,3,6}}; // matrix A
    int B[K][N] = {{2,3}, {4,5}, {6,7}}; // matrix B
    int C[M][N] = {0}; // result matrix

    int *d_A, *d_B, *d_C; // device pointers for the matrices
    cudaMalloc((void **)&d_A, M * K * sizeof(int)); // allocate memory for the matrices on the device
    cudaMalloc((void **)&d_B, K * N * sizeof(int));
    cudaMalloc((void **)&d_C, M * N * sizeof(int));

    dim3 blockSize(16, 16); // block size for the kernel
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y); // grid size for the kernel

    cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice); // copy the matrices from the host to the device
    cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice); 

    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N); // launch the kernel
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost); // copy the result from the device to the host

    for (int i = 0; i < M; i++)  // print the result
    {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    cudaFree(d_A); // free the memory for the matrices on the device
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}




