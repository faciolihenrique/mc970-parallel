#include <stdio.h>

__global__ void add(int *a, int *b, int *c) { 
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; 
}

int main(void) {
	int a[4] = {2,2,2,2};
	int b[4] = {7,7,7,7};
	int c[4];
	int *d_a, *d_b, *d_c;
	int size = 4*sizeof(int);

	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<4,1>>>(d_a, d_b, d_c); 

	cudaMemcpy(c, d_c, size,cudaMemcpyDeviceToHost);

	printf("%d %d %d %d\n", c[0], c[1], c[2], c[3]);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0; 
}
