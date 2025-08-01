

"""
author: Hadi Vareno
e-mail: mohammad.noori.vareno@uni-jena.de
github: https://github.com/TheVareno
"""
 
from read5.Reader import read # type: ignore 
from ont_fast5_api.conversion_tools.fast5_subset import Fast5Filter # type: ignore 
# import ont_fast5_api # type: ignore 
import argparse
# TODO use hampelFilter from FileIO.py
from hampel import hampel # type: ignore
import subprocess as sp
import multiprocessing as mp
import queue
import os   
from typing import List
from pathlib import Path 
import pysam # type: ignore
 
def get_read_data(input_path: str) -> any:
    
    allowed_extensions = {'fast5', 'pod5', 'slow5'}

    if os.path.isfile(input_path):
        extention = input_path.split('.')[1]  
        if extention in allowed_extensions:
            return input_path
        else: 
            raise ValueError(f"Given file format not acceptable, allowd formats: FAST5, POD5 or SLOW5")
        
    if os.path.isdir(input_path):
        res_files = []
        for file_path in os.listdir(input_path): 
            if os.path.isfile(os.path.join(input_path, file_path)): 
                extention = os.path.join(input_path, file_path).split('.')[1] 
                if extention in allowed_extensions: 
                    res_files.append(file_path)
                
                else:
                    raise ValueError(
                    f"Working directory does not contain any FAST5, POD5, or SLOW5 read data files.")

        return res_files
    


"""
- takes basecalled bam file  
- calcultes average samples per nucleotide : ns / read_length_nt 
- searches in given basecalled file in bam format for read lenght in 
- returns the a dict of all read ids as key, sampling rate as value    
"""
def calculate_sample_rate(read_id: str, bam_file: str, mode='rb')-> dict: 

    samfile = pysam.AlignmentFile(bam_file, mode, check_sq=False) 
    read_id_length_pairs = {} 
    
    for read in samfile.fetch(until_eof=True): 
        
        read_name = read.query_name
        # tags = dict(read.tags) 
        
        read_nt_length = int(read.query_length)
        ns = int(dict(read.tags['ns'])) 
        read_avg_sig_per_nt = ns / read_nt_length 

        read_id_length_pairs.update({read_name: read_avg_sig_per_nt})
    
    return read_avg_sig_per_nt

def find_polya(task_queue: mp.Queue, result_queue: mp.Queue, input_file: str): 

    read_object = read(input_file)

    while True:
        try:
            read_id = task_queue.get_nowait() 
 
            z_normalized_signal_values = read_object.getZNormSignal(read_id, mode='mean')
            filter_object = hampel(z_normalized_signal_values, window_size=5, n_sigma=6.0)
            filtered_signal_values = filter_object.filtered_data
            
            if len(filtered_signal_values) == 0:
                print(f"[WARN] Empty filtered signal for read: {read_id}")
                continue
            
            sig_vals_str = ','.join(map(str, filtered_signal_values))
            
            with open('signal.txt', 'w') as f:
                f.write(sig_vals_str)

            if not sig_vals_str: 
                print(f"[WARN] Empty signal values for read {read_id}")

            if any(val in sig_vals_str for val in ['nan', 'inf', '-inf']):
                 print(f"[DEBUG] WARNING: 'nan' or 'inf' found in signal string for read {read_id}. This might cause issues for polyA.")

            process = sp.Popen(['./polyA'], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            process.stdin.write(f"{sig_vals_str}\n") 
            process.stdin.flush()
            stdout, stderr = process.communicate()
            rc = process.returncode # int 
            
            if rc == 0:  
                result_queue.put((read_id, stdout.strip()))
                print(f"borders recieved! -->> {rc}")
            else: 
                print(f"[ERROR] polyA finder exited with code {rc} for read {read_id}")
                if stderr != '': 
                    print(f"[STDERR] Error for {read_id}: {stderr}")

        except queue.Empty:
            break



def start_finder(input_file, output_path, sample_rates: dict):
    
    if not os.path.exists(output_path):  
        os.makedirs(output_path) 
    
    save_file = os.path.join(output_path, f'output_test.csv')

    with open(save_file, 'w') as f: # file exist. check 
        f.write("Read ID, poly(A) start, poly(A) end, poly(A) estimated length\n")
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    read_object = read(input_file)         
    all_read_ids = read_object.getReads() 

    for r_id in all_read_ids:
        task_queue.put(r_id)

    num_processes = os.cpu_count() 
    
    processes = [mp.Process(target=find_polya, args=(task_queue, result_queue, input_file))
         for _ in range(num_processes)]
    
    for process in processes:
        process.start()

    for process in processes:
        process.join()



    #results = []
    while not result_queue.empty():
        
        #results.append(result_queue.get())
        
        read_id, borders = result_queue.get() 
        polyA_estimated_lenght = (int(borders[1]) - int(borders[0])) / sample_rates[read_id]

        with open (save_file, 'a') as f: 
            f.write(f"{read_id},{borders[0]},{borders[1]},{polyA_estimated_lenght}\n")
    
    #return results



def split_segment_input(input_read_data: str, output_path: str, summary_file_path: str):

    #name_read_data = input_read_data.split()[0]
        
    if not os.path.exists(output_path):  
        os.makedirs(output_path) 
    
    save_file = os.path.join(output_path, f'output_test.csv')

    with open(save_file, 'w') as f: # file exist. check 
        f.write("Read ID, poly(A) end, adapter end, leader end, start end\n")
    
    """
    with open(save_file, 'w') as f: # file exist. check 
        f.write("Read ID, poly(A) start, poly(A) end, poly(A) estimated length \n")
    """

    splitter = Fast5Filter(
                input_folder=input_read_data, 
                output_folder=output_path, 
                read_list_file=summary_file_path,
                filename_base="subset",
                batch_size=500, 
                threads=1,
                recursive=False,
                file_list_file=None,
                follow_symlinks=False,
                target_compression=None)

    splitter.run_batch() 

    for file in os.listdir(output_path): 

        if file.endswith(".fast5") or file.endswith(".pod5") or file.endswith(".slow5"): 
            
            file = os.path.join(output_path, file)
            read_object = read(file) # needs file path ends with .fast5 / .pod5 / .slow5
            
            all_read_ids = read_object.getReads() # 500 each time
            
            task_queue = mp.Queue()
            result_queue = mp.Queue()

            for read_id in all_read_ids:
                task_queue.put(read_id)
    
            number_of_processes = os.cpu_count()
            
            processes = [mp.Process(target=find_polya, args=(task_queue, result_queue, file)) 
                         for _ in range(number_of_processes)]
    
            for proc in processes:
                proc.start()
        
            for proc in processes:
                proc.join()
    
            while not result_queue.empty():
                read_id, borders_length = result_queue.get()
                with open (save_file, 'a') as f: 
                    f.write(f"{read_id},{borders_length}\n")
        else: 
            continue



def main(): 

    parser = argparse.ArgumentParser(description="Process and Save output file.")
    parser.add_argument("--input_dir", 
                        type=str, required=True, 
                        help="Path to directory containing input ONT read data in FAST5, POD5, or SLOW5 format.")
    
    parser.add_argument("--output_dir", 
                        type=str, required=True, 
                        help="Directory to save output files.")
    
    parser.add_argument("--bam_file", 
                        type=str, required=False, 
                        help="Path to basecalled bam file.")
    
    args = parser.parse_args()

    input_path = get_read_data(args.input_dir)

    read_id_sample_rate_pairs = calculate_sample_rate(args.bam_file)

    if isinstance(input_path, str): 
        start_finder(input_path, args.output_dir, read_id_sample_rate_pairs) 
    else:  
        for read_file in input_path: 
            start_finder(read_file, args.output_dir, read_id_sample_rate_pairs) 
    
    
    # split_segment_input(args.input_dir, args.output_dir, args.summary_file)        


    #try:
        # read_data_files = get_read_data(args.input_dir)  
        
        #for read_file in read_data_files:

    #except FileNotFoundError as e:
    #    print(f"Error: {e}")
    #    print("Please ensure the directory exists and the path is correct.")
    #except ValueError as e:
    #    print(f"Error: {e}")
    #    print("Please ensure the directory contains .fast5, .pod5, or .slow5 files.")
    #except Exception as e: 
    #    print(f"An unexpected error occurred: {e}")


    
    


if __name__ == '__main__' : 
    main()






"""
with open(save_file, 'w') as f: # file exist. check 
    f.write("Read ID, poly(A) start, poly(A) end, poly(A) estimated length \n")
"""
    