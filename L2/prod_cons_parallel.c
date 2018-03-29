/**
 * Henrique Noronha Facioli
 * 157986
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* This function was previously provided. Just changed it to not receive buff */
void producer_consumer(int size_of_buffer, int *vec, int n, int n_threads) {
	int buffer[size_of_buffer];			// Declared as this to be declared on every
	int i, j;												// iteration of omp for thread
	long long unsigned int sum = 0;

	// As the consumer only gets from the buffer from i-1, we can buffer private
	//  and iterate 2 by 2 and we'll never get any interference
	//
	#	pragma omp parallel for \
			num_threads(n_threads) \
			default(none) \
			shared(vec, n, size_of_buffer) \
			private(buffer, i, j) \
			reduction(+: sum) \
			schedule(static, 2)
	for (i = 0; i < n; i++) {
		if(i % 2 == 0) {	// PRODUTOR
			for(j=0;j<size_of_buffer;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMIDOR
			for(j=0;j<size_of_buffer;j++) {
				sum += buffer[j];
			}
		}
	}
	printf("%llu\n",sum);
}

int main() {
	int i;
	int n_threads, n_loop_iterations, buffer_size, read;
	int *read_vector;
	double start, end;

	scanf("%d", &n_threads );
	scanf("%d", &n_loop_iterations );
	scanf("%d", &buffer_size );

	read_vector = (int*)calloc(n_loop_iterations, sizeof(int));


	for(i = 0; i < n_loop_iterations; i++) {
		scanf("%d", &read);
		read_vector[i] = read;
	}

	start = omp_get_wtime();
	producer_consumer(buffer_size, read_vector, n_loop_iterations, n_threads);
	end = omp_get_wtime();

	free(read_vector);

	printf("%lf\n", end - start);
	return 0;
}
