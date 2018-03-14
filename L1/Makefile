CC = gcc
FLAGS = -g -Wall -fopenmp

all: t1_seq t1_par

t1_seq: T1-seq.c
  $(CC) T1-seq.c $(FLAGS) -o T1-seq.bin

t1_par: T1-par.c
  $(CC) T1-par.c $(FLAGS) -o T1-par.bin
