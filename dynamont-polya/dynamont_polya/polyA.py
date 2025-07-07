
"""
author: Hadi Vareno
e-mail: mohammad.noori.vareno@uni-jena.de
github: https://github.com/TheVareno
"""

import numpy as np #type: ignore
from read5.Reader import read # type: ignore 
import ont_fast5_api.fast5_interface     # type: ignore 
import argparse
from hampel import hampel # type: ignore
import subprocess as sp
import os  
import multiprocessing as mp 
import queue

def setup_working_directory():     
    if os.getcwd() != '/home/hi68ren/Dokumente/ProjektModul/Implementation/scripts/dynamont_polya' : 
        os.chdir('/home/hi68ren/Dokumente/ProjektModul/Implementation/scripts/dynamont_polya')
    else : 
        pass
    
#---------------- SEGMENTATION SECTION -----------------        

def find_polya(task_queue: mp.Queue, result_queue: mp.Queue, input_path: str): 
    
    read_object = read(input_path) 
    
    while not task_queue.empty(): 
        try:
            read_id = task_queue.get_nowait() 
            z_normalized_signal_values = read_object.getZNormSignal(read_id, mode='mean')
            filter_object = hampel(z_normalized_signal_values, window_size=5, n_sigma=6.0)
            filtered_signal_values = filter_object.filtered_data

            if len(filtered_signal_values) == 0:
                print(f"the array of signal values empty for read id : {read_id}")
                
            polyA_app_call = ['./polyA']  
            
            sig_vals_str = ','.join(map(str, filtered_signal_values))
            
            process = sp.Popen(polyA_app_call, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            
            if not sig_vals_str: 
                print(f"Empty signal values for read {read_id}")
            
            process.stdin.write(f"{sig_vals_str}\n")
            process.stdin.flush()
            stdout, stderr = process.communicate()
            rc = process.returncode # int 
            
            if rc == 0:  
                borders = stdout.strip()
                result_queue.put((read_id, borders))
            else: 
                # print(f"no borders for read id: {read_id}")
                pass
            
            if stderr: 
                print(f"Error for {read_id}: {stderr}")
                continue

        except queue.Empty:
            break
            
def main(): 
    
    setup_working_directory() 

    parser = argparse.ArgumentParser(description="Process and Save output file.")
    parser.add_argument("--input_file", type=str, help="Path to ONT read data in either FAST5, POD5 or SLOW5.")
    parser.add_argument("--output_path", type=str, help="Path to save output file.")
    args = parser.parse_args()

    file_path = args.input_file
    
    save_file = os.path.join(args.output_path, 'region_borders.txt') 
    
    if not os.path.isfile(file_path): # FAST5/POD5/SLOW5 is there ? 
        raise FileNotFoundError(f'The input file {args.input_file} not found.')

    if not os.path.exists(args.output_path): # dir exist. check 
        os.makedirs(args.output_path)
    
    with open(save_file, 'w') as f: # file exist. check 
        f.write("Read ID,Poly(A) end,Adapter end,Leader end,Start end\n")
    
    read_object = read(file_path)
    all_read_ids = read_object.getReads() 
    print(f"Number of all reads to find poly(A) in : {len(all_read_ids)} reads")
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for read_id in all_read_ids:
        task_queue.put(read_id)
    
    number_of_processes = os.cpu_count()
    print(f"Number of available processes : {number_of_processes}")
        
    processes = [ mp.Process(target=find_polya, args=(task_queue, result_queue, file_path)) for _ in range(number_of_processes) ]
    
    for proc in processes:
        proc.start()
        
    for proc in processes:
        proc.join()
    
    while not result_queue.empty():
        read_id, borders = result_queue.get()
        with open (save_file, 'a') as f: 
            f.write(f"{read_id},{borders}\n")



if __name__ == '__main__' : 
    main()






