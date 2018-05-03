/*
 * Henrique Noronha Facioli
 * 157986
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void add(int *a, int *b, int *c, int *line_length, int *row_length) {
    // Based on slide 6 from class
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < *line_length && j < *row_length) {
        int pos = i*(*row_length)+j;
        c[pos] = a[pos] + b[pos];
    }
}

int main()
{
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int i, j;

    //Input
    int linhas, colunas;
    int *d_linhas, *d_colunas;

    scanf("%d", &linhas);
    scanf("%d", &colunas);

    //Alocando memória na CPU
    A = (int *)malloc(sizeof(int)*linhas*colunas);
    B = (int *)malloc(sizeof(int)*linhas*colunas);
    C = (int *)malloc(sizeof(int)*linhas*colunas);

    // Alocate space for all the variables on GPU memory
    cudaMalloc((void **) &d_A, sizeof(int)*linhas*colunas);
    cudaMalloc((void **) &d_B, sizeof(int)*linhas*colunas);
    cudaMalloc((void **) &d_C, sizeof(int)*linhas*colunas);
	cudaMalloc((void **) &d_linhas, sizeof(int));
	cudaMalloc((void **) &d_colunas, sizeof(int));
    
    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    // Send the matrix and length values to GPU
    cudaMemcpy(d_A, A, sizeof(int)*linhas*colunas, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int)*linhas*colunas, cudaMemcpyHostToDevice);
    cudaMemcpy(d_linhas, &linhas, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colunas, &colunas, sizeof(int), cudaMemcpyHostToDevice);

    // Call the computation using 2d
    // https://codeyarns.com/2011/02/16/cuda-dim3/ and 
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3
    int nlin = ceil((float)colunas / 32);     // Create the total number of blocs necessary
    int nrow = ceil((float)linhas / 32);      // to iterate through matrix of size 32
    dim3 gridDim(nlin, nrow);                 // Now, iterate on a grid at most 32x32
    dim3 blockDim(32, 32);                    //  32 x 32 x 1 (1024)

    add<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_linhas, d_colunas);

    // Gets from the device the values
    cudaMemcpy(C, d_C, sizeof(int)*linhas*colunas, cudaMemcpyDeviceToHost);

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);

    // Free the memory allocated to the process
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_linhas);
    cudaFree(d_colunas);

    free(A);
    free(B);
    free(C);
}

