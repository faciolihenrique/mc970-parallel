#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* count sort serial */
double count_sort_serial(double a[], int n) {
	int i, j, count;
	double *temp;
	double start, end, duracao;

	temp = (double *)malloc(n*sizeof(double));

	start = omp_get_wtime();
	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
		temp[count] = a[i];
	}
	end = omp_get_wtime();

	duracao = end - start;

	memcpy(a, temp, n*sizeof(double));
	free(temp);

	return duracao;
}

int main(int argc, char * argv[]) {
	int i, n, nt;
	double  * a, t_s;

	scanf("%d",&nt);
	
	/* numero de valores */
	scanf("%d",&n);

	/* aloca os vetores de valores para o teste em serial(b) e para o teste em paralelo(a) */
	a = (double *)malloc(n*sizeof(double));

	/* entrada dos valores */
	for(i=0;i<n;i++)
		scanf("%lf",&a[i]);
	
	/* chama as funcoes de count sort em paralelo e em serial */
	t_s = count_sort_serial(a,n);
	
	/* Imprime o vetor ordenado */
	for(i=0;i<n;i++)
		printf("%.2lf ",a[i]);

	printf("\n");

	/* imprime os tempos obtidos e o speedup */
	printf("%lf\n",t_s);

	return 0;
}
