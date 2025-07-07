#!/bin/bash 

# main command to run the app in train mode 
# init. with nnps parameters 
./polyA -s 0.1 -l1 0.9 -l2 0.9 -a1 0.1 -a2 0.95 -pa1 0.05 -pa2 0.9 -tr1 0.1 -tr2 0.99 -s 0.01 < test_sig_vals.txt
    