/**
 * Henrique Noronha Facioli
 * 157986
 *
 *	Trabalho complementar
 *  Coleta dos dados
 *  Para o código serial
 * 		-arq1: 1169
 *		-arq2: 32971
 * 		-arq3: 255882
 *  Para o código paralelo temos:
 *NThreads|   1    |    2    |    4    |    8    |    16    |
 *	-arq1:| 1385	 |   536	 |   291	 |   333	 |   759		|
 *	-arq2:| 35032	 |  19533	 |  12000	 |  6025	 |   5140		|
 *	-arq3:| 253050 |  136579 |  73844	 |  44275  |  46572   |
 *	|----------------------------------------------|
 *	|		 | N_Threads|  1  |  2  |  4  |  8  |  16  |
 *	|arq1|Speed Up	|0.84 |2.18 |4.01 |3.51 |1.54  |
 *	|		 |Eficiencia|0.84 |1.09 |1.00 |0.43 |0.09  |
 *	|arq2|Speed Up	|0.94 |1.72 |2.74 |5.77 |6.41  |
 *	|		 |Eficiencia|0.94 |0.86 |0.68 |0.72 |0.40  |
 *	|arq3|Speed Up	|1.01 |1.87 |3.46 |5.47 |5.49  |
 *	|		 |Eficiencia|1.01 |0.98 |0.86 |0.68 |0.34  |
 *	|----------------------------------------------|
 * Como podemos ver, ao compararmos um código serial com um paralelo rodando
 * uma única thread, teremos o código serial mais rápido ou equivalente ao para-
 * lelo. Isso se deve ao fato de ao rodarmos um programa paralelo existe todo o
 * gasto para acordar a thread e pará-la, que não acontece em um código serial.
 * Já com duas threads, podemos ver que há uma melhora de quase 100% em todos os
 * testes, o que, ao considerar que o processador está executando outros proces-
 * sos e threads paralelamente ao programa, podemos desconsiderar um certo erro
 * e assumir um ganho de 2x com eficiencia de quase 100%. O mesmo acontece para
 * quatro threads que possui seu desempenho quase o dobro de 2 threads, tendo um
 * rendimento maior que 68% em todos os casos.
 * No entanto, o paralelismo começa a não ter o mesmo efeito a partir de 8 e 16
 * threads, onde para o arq1, por ser de tamanho pequeno, o gerencimento de th-
 * reads já leva mais tempo que o próprio programa e para os tests arq2 e 3, os
 * tempos começam a se manter estáveis.
 * Utilizando a lei de Amdahl podemos ver que este código está aproximadamente
 * 20% paralelizavel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

pthread_mutex_t shared_sum_operation = PTHREAD_MUTEX_INITIALIZER;

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

	// Here we can parelallize the external or the internal loop
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
		// Protect the shared variable to not occurr two sum simultaneously :-)
		pthread_mutex_lock(&shared_sum_operation);
		vet[j] += count;
		pthread_mutex_unlock(&shared_sum_operation);
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
	double constant_multiplier = ((double) n_reads/(double) n_threads);	// Used many times
	h = (max_read - min_read)/n_bins;
	pthread_arg **args = (pthread_arg**)calloc(n_threads, sizeof(pthread_arg*));
	for (i = 0; i < n_threads; i++)
		args[i] = (pthread_arg*)calloc(n_threads, sizeof(pthread_arg));

	gettimeofday(&start, NULL);
	// Execute the threads
	for (i = 0; i < n_threads; i++) {
		args[i]->h = h;
		args[i]->max = max_read;
		args[i]->min = min_read;
		args[i]->nbins = n_bins;
		args[i]->nval_begin = ceil(i*constant_multiplier);
		args[i]->nval_end = ceil((i+1)*constant_multiplier-1);
		args[i]->val = read_vector;
		args[i]->vet = bin_vec;

		pthread_create(&thread_handlers[i], NULL, Count, args[i]);
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


	// Printing the output
	for(i = 0; i < n_bins; i++){
		printf("%.2lf ", min_read + i*h);
	}
	printf("%.2f\n", min_read + n_bins*h);
	for (i = 0; i < n_bins-1; i++) {
		printf("%d ", bin_vec[i]);
	}
	printf("%d\n", bin_vec[n_bins-1]);
	printf("%lu\n", duracao);

	// Free memory
	for(i = 0; i < n_threads; i++ )
		free(args[i]);
	free(args);
	free(read_vector);
	free(bin_vec);
	free(thread_handlers);

	return 0;
}
