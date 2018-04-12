#!/bin/bash
for i in `seq 1 10`;
do
	echo $i
	echo "./senha-serial.bin < arq$i.in"
	./senha-serial.bin < arq$i.in
        echo "./senha-openmp.bin < arq$i.in"
        ./senha-openmp.bin < arq$i.in

done    
