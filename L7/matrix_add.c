#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
    int *A, *B, *C;
    int i, j;

    //Input
    int linhas, colunas;

    scanf("%d", &linhas);
    scanf("%d", &colunas);

    //Alocando memória na CPU
    A = (int *)malloc(sizeof(int)*linhas*colunas);
    B = (int *)malloc(sizeof(int)*linhas*colunas);
    C = (int *)malloc(sizeof(int)*linhas*colunas);
    
    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    //Computacao que deverá ser movida para a GPU (que no momento é executada na CPU)
    //Lembrar que é necessário usar mapeamento 2D (visto em aula) 
    for(i=0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            C[i*colunas+j] = A[i*colunas+j] + B[i*colunas+j];
        }
    }

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);

    free(A);
    free(B);
    free(C);
}

