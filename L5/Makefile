# Makefile for md5

TARGET=md5
TARGET_PAR=md5_par
OBJS=md5.o md5_bmark.o
OBJS_PAR=md5.o md5_bmark-par.o
CFLAGS=-O2 -g -D_GNU_SOURCE -std=c99
LDFLAGS= -fopenmp
CC=gcc

all: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(TARGET) -fopenmp

par: $(OBJS_PAR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS_PAR) -o $(TARGET_PAR) -fopenmp

%.o : %.c
	$(CC) $(CFLAGS) -c $< -fopenmp

clean:
	rm -f *.o $(TARGET) $(TARGET_PAR) output.txt
