/**
 * Henrique Noronha Facioli
 * 157986
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

pthread_mutex_t shared_sum_operation = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
	unsigned int thread_n;
	long long int begin_loop;
	long long int end_loop;
	long long int *sum_in_circle;
} pthread_arg;

void* Throw_darts(void* args) {
	pthread_arg *p_arg = (pthread_arg*)args;
	unsigned int thread_n = p_arg->thread_n;
	long long int begin_loop = p_arg->begin_loop;
	long long int end_loop = p_arg->end_loop;
	long long int *sum_in_circle = p_arg->sum_in_circle;

	double x, y;
	long long int i, darts_in_circle = 0;
	unsigned int seed1 = (thread_n+50)*rand();
	unsigned int seed2 = (thread_n+100)*rand();

	for (i = begin_loop; i < end_loop; i++){
		x = -1 + 2*(double)rand_r(&seed1)/(double)((unsigned)RAND_MAX + 1);
		y = -1 + 2*(double)rand_r(&seed2)/(double)((unsigned)RAND_MAX + 1);

		if ((x*x + y*y) < 1.0) {
			darts_in_circle += 1;
		}
	}

	pthread_mutex_lock(&shared_sum_operation);
	*sum_in_circle += darts_in_circle;
	pthread_mutex_unlock(&shared_sum_operation);

	return p_arg;
}

/** main **/
int main() {
	unsigned int n_threads, n_darts, i;
	pthread_t* thread_handlers;
	long long int *sum = (long long int*)calloc(1, sizeof(long long int));
	struct timeval start, end;

	scanf("%u", &n_threads);
	scanf("%u", &n_darts);

	// Allocate a vector of threads to be executed
	thread_handlers = (pthread_t*)calloc(n_threads, sizeof(pthread_t));

	pthread_arg **args = (pthread_arg**)calloc(n_threads, sizeof(pthread_arg*));
	for (i = 0; i < n_threads; i++)
		args[i] = (pthread_arg*)calloc(n_threads, sizeof(pthread_arg));

	gettimeofday(&start, NULL);
	// Execute the threads
	for (i = 0; i < n_threads; i++) {
		args[i]->thread_n = i;
		args[i]->begin_loop = i*(n_darts/n_threads);
		args[i]->end_loop = (i+1)*(n_darts/n_threads);
		args[i]->sum_in_circle = sum;
		pthread_create(&thread_handlers[i], NULL, Throw_darts, args[i]);
	}

	// Wait all the threads to execute
	for (i = 0; i < n_threads; i++) {
		pthread_join(thread_handlers[i], NULL);
	}
	gettimeofday(&end, NULL);


	// Calculate the durationg based on formula given on exercice
	unsigned long int duracao = (
		(end.tv_sec*1000000 + end.tv_usec) -(start.tv_sec*1000000 + start.tv_usec)
	);

	printf("%lf\n", 4.0*(double)*sum/(double)n_darts);
	printf("%lu\n", duracao);

	// Free memory
	for(i = 0; i < n_threads; i++ )
		free(args[i]);
	free(args);
	free(thread_handlers);

	return 0;
}
