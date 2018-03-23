/**
 * Henrique Noronha Facioli
 * 157986
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

typedef struct {
	double min;			// Minimum value for the bin containing the smallestvalues
	double max;  		// The maximum value for the bin containing the largest values
	int nbins;			// The number of bins
	int nval_begin;	// First item to start iteration on array
	int nval_end;		// Last item to end iteration on array
	double h;				// Each bin width
	double * val; 	// The array of input data
	int *vet;				// An array of bin count ints
} pthread_arg;

void* Count(void* args) {
	int i, j, count;
	double min_t, max_t;

	// Extract the information from argumentos
	pthread_arg* this_pthread_arg = (pthread_arg*) args;
	double min = this_pthread_arg->min;
	int nbins = this_pthread_arg->nbins;
	int nval_begin = this_pthread_arg->nval_begin;
	int nval_end = this_pthread_arg->nval_end;
	double h = this_pthread_arg->h;
	double *val = this_pthread_arg->val;
	int *vet = this_pthread_arg->vet;

	// printf("min : %lf\n", min);

	for (j = 0; j < nbins; j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;
		for (i = nval_begin; i <= nval_end; i++) {
			if( (val[i] <= max_t && val[i] > min_t) ||
					(j == 0 && val[i] <= min_t)
				) {
				count++;
			}
		}
		vet[j] += count;
	}

	return vet;
}

/** main **/
int main() {
	int i;
	int n_threads, n_reads, n_bins;
	pthread_t* thread_handlers;
	int *bin_vec;
	double *read_vector;
	double max_read, min_read, h;
	struct timeval start, end;

	scanf("%d", &n_threads);
	scanf("%d", &n_reads);
	scanf("%d", &n_bins);

	// Allocate the vector for reading and a vector of threads to be executed
	read_vector = (double*)calloc(n_reads, sizeof(double));
	bin_vec = (int*)calloc(n_bins, sizeof(double));
	thread_handlers = (pthread_t*)calloc(n_threads, sizeof(pthread_t));

	// Read the numbers from stdin
	min_read = 99999999.0;				// Sets a min/max value
	max_read = -99999999.0;				// Sets a min/max value
	for(i = 0; i < n_reads; i++) {
		scanf("%lf", &read_vector[i]);

		// Find the minimum value on reading
		if (read_vector[i] < min_read) {
			min_read = read_vector[i];
		}

		// Find the maximum value on reading
		if (read_vector[i] > max_read) {
			max_read = read_vector[i];
		}
	}

	min_read = floor(min_read);			// Need this to obtain the same answer as res
	max_read = ceil(max_read);			// Need this to obtain the same answer as res
	h = (max_read - min_read)/n_bins;

	pthread_arg *args = (pthread_arg*)calloc(n_threads, sizeof(pthread_arg));

	gettimeofday(&start, NULL);
	args->h = h;
	args->max = max_read;
	args->min = min_read;
	args->nbins = n_bins;
	args->nval_begin = 0;
	args->nval_end = n_reads-1;
	args->val = read_vector;
	args->vet = bin_vec;

	Count(args);
	gettimeofday(&end, NULL);


	// Calculate the durationg based on formula given on exercice
	double duracao = (
		(end.tv_sec*1000000 + end.tv_usec) -(start.tv_sec*1000000 + start.tv_usec)
	);


	// Printing the output
	for(i = 0; i < n_bins; i++){
		printf("%.2lf ", min_read + i*h);
	}
	printf("%.2f\n", min_read + n_bins*h);
	for (i = 0; i < n_bins-1; i++) {
		printf("%d ", bin_vec[i]);
	}
	printf("%d\n", bin_vec[n_bins-1]);
	printf("%.lf\n", duracao);

	// Free memory
	free(args);
	free(read_vector);
	free(bin_vec);
	free(thread_handlers);

	return 0;
}
