#!/usr/bin/env bash

./sequence_learning.py -u 0 -n u0_a005 -p 100 -e 25000 -a 0.05 &
./sequence_learning.py -u 0 -n u0_a001 -p 100 -e 25000 -a 0.01 &
./sequence_learning.py -u 0 -n u0_a02 -p 100 -e 25000 -a 0.2 &





