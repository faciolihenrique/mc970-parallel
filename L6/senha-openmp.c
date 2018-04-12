/* Henrique Noronha Facioli
 * RA: 157986
 * Lab 6 - Quebrar senha arquivo ZIP
 * Para está abordagem o será usado openmp tasks em que será criada uma fila
 * de tamanho 500000 e cada uma das taskas será encarregada de executar a aber-
 * tura e leitura do arquiv (parte mais demorada). Será utilizada uma variável
 * shared entre as threads que impedem que ela execute o código caso já tenham
 * encontrado a senha
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

FILE *popen(const char *command, const char *type);

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


int main ()
{
  FILE * fp;
  char finalcmd[300] = "unzip -P%d -t %s 2>&1";
  int nt;

  char filename[100];
  char ret[200];
  char cmd[400];
  double t_start, t_end;

  int i;
  scanf("%d", &nt);
  scanf("%s", filename);

  t_start = rtclock();

  char nao_achou = 1;
  #pragma omp parallel num_threads(nt) \
    private(cmd, i, fp, ret) \
    shared(nao_achou, finalcmd, filename)
  #pragma omp single nowait
  for (i=0; i < 500000 && nao_achou; i++) {

    #pragma omp task
    if (nao_achou){
      sprintf((char*)&cmd, finalcmd, i, filename);

      fp = popen(cmd, "r");
      while (!feof(fp)) {
        fgets((char*)&ret, 200, fp);
        if (strcasestr(ret, "ok") != NULL) {
          printf("Senha:%d\n", i);
          i = 500000;
          nao_achou = 0;
        }
      }
      pclose(fp);
    }
  }
  #pragma omp taskwait
  t_end = rtclock();
  fprintf(stdout, "%0.6lf\n", t_end - t_start);
}
