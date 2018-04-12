CC = gcc
FLAGS = -g -fopenmp -Wall -pedantic -std=c99 -lm

all: senha-serial senha-parallel

senha-serial: senha-serial.c
	$(CC) senha-serial.c $(FLAGS) -o senha-serial.bin

senha-parallel: senha-openmp.c
	$(CC) senha-openmp.c $(FLAGS) -o senha-openmp.bin

clean:
	rm -rf senha-openmp.bin senha-serial.bin
