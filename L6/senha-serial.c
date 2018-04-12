#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

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
   for(i=0; i < 500000; i++){
	sprintf((char*)&cmd, finalcmd, i, filename);
	//printf("Comando a ser executado: %s \n", cmd); 

        fp = popen(cmd, "r");	
	while (!feof(fp)) {
		fgets((char*)&ret, 200, fp);
		if (strcasestr(ret, "ok") != NULL) {
			printf("Senha:%d\n", i);
			i = 500000;
		}
	}
	pclose(fp);
  }
  t_end = rtclock();
 
  fprintf(stdout, "%0.6lf\n", t_end - t_start);  
}
