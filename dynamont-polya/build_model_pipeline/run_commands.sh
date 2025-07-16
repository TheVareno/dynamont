#!bin/bash 

# main command to run the app in train mode 
# init. with nnps parameters 
call_polya_cpp_train(){
    ./polyA -s 0.1 -l1 0.9 -l2 0.9 -a1 0.1 -a2 0.95 -pa1 0.05 -pa2 0.9 -tr1 0.1 -tr2 0.99 -s 0.01 < test_sig_vals.txt
} 


call_reg_cmd() {
    echo 'S: 0.1; L1: 0.9; L2: 0.9; A1: 0.1; A2: 0.95; PA1: 0.05; PA2: 0.9; TR1: 0.1; TR2: 1.0' > nnps_initial_parameters.txt
    >> training_read_ids.txt 
    > ttraining_signal_values.txt   
}


update_parameters(){
    # reset the using parmater 
    echo 'S: 0.1; L1: 0.9; L2: 0.9; A1: 0.1; A2: 0.95; PA1: 0.05; PA2: 0.9; TR1: 0.1; TR2: 1.0' > updated_parameters.txt
}

feature="$1"

if [[ "$feature"== "update"]]; then 
    update_parameters 
elif [[ "$feature"== "reg"]]; then
    call_polya_cpp_train 
else 
    call_polya_cpp_train