

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
from pathlib import Path
from typing import List
import queue
import os  

def get_read_data(input_path: str) -> List[Path]:
    
    allowed_extensions = {'.fast5', '.pod5', '.slow5'}

    if os.path.isfile(input_path):
        extention = input_path.split('.')[1]  
        if extention in allowed_extensions:
            return input_path
        else: 
            raise ValueError(f"Given file format not acceptable, allowd formats: FAST5, POD5 or SLOW5")
    
    
    if os.path.isdir(input_path):
        input_dir = Path(input_path)

        found_files = [
            file_path for file_path in input_path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
        ]

        if not found_files:
            raise ValueError(
                f"Working directory '{input_dir}' does not contain any "
                "FAST5, POD5, or SLOW5 read data files."
            )

        return found_files
    

def find_polya(task_queue: mp.Queue, result_queue: mp.Queue, read_object: str): 
    
    while not task_queue.empty(): 
        try:
            read_id = task_queue.get_nowait() 
            z_normalized_signal_values = read_object.getZNormSignal(read_id, mode='mean')
            filter_object = hampel(z_normalized_signal_values, window_size=5, n_sigma=6.0)
            filtered_signal_values = filter_object.filtered_data

            if len(filtered_signal_values) == 0:
                print(f"the array of signal values empty for read id : {read_id}")
                
            polyA_app_call = './polyA'  
            
            sig_vals_str = ','.join(map(str, filtered_signal_values))
            
            process = sp.Popen(polyA_app_call, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            
            if not sig_vals_str: 
                print(f"Empty signal values for read {read_id}")
            
            process.stdin.write(f"{sig_vals_str}\n")
            process.stdin.flush()
            stdout, stderr = process.communicate()
            rc = process.returncode # returns int 
            
            if rc == 0:  
                borders = stdout.strip()
                result_queue.put((read_id, borders))
            else: 
                pass
            
            if stderr: 
                print(f"Error for {read_id}: {stderr}")
                continue

        except queue.Empty:
            break


"""
- raw signal splitting into 8 files of 500 reads 
"""
def split_segment_input(input_read_data: str, output_path: str, summary_file_path: str):

    name_read_data = input_read_data.split()[0]
        
    if not os.path.exists(output_path):  
        os.makedirs(output_path) 
    
    save_file = os.path.join(output_path, f'output_{name_read_data}.csv')

    with open(save_file, 'w') as f: # file exist. check 
        f.write("Read ID, poly(A) end, adapter end, leader end, start end\n")
    
    # alternative - file handling 
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
            
            processes = [ mp.Process(target=find_polya, args=(task_queue, result_queue, read_object)) 
                         for _ in range(number_of_processes) ]
    
            for proc in processes:
                proc.start()
        
            for proc in processes:
                proc.join()
    
            while not result_queue.empty():
                read_id, borders = result_queue.get()
                with open (save_file, 'a') as f: 
                    f.write(f"{read_id},{borders}\n")
        else: 
            continue



def main(): 
    # TODO use same parameters/namings as dynamont 

    parser = argparse.ArgumentParser(description="Process and Save output file.")
    parser.add_argument('-i', "--input_dir", type=str, required=True, help="Path to directory containing input ONT read data in FAST5, POD5, or SLOW5 format.")
    parser.add_argument('-o', "--output_dir", type=str, required=False, help="Directory to save output files.")
    parser.add_argument('-s', "--summary_file", type=str, required=False, help="Path to the sequence summary file.")
    parser.add_argument('-t', "--test", type=str)

    args = parser.parse_args()

     
    try:
        read_data_files : List[Path] = get_read_data(args.input_dir) # list obj: contains all reads 
        
        for read_data_file in read_data_files: 
            
            # print(read_data_file) 
            split_segment_input(input_read_data=read_data_file, output_path=args.output_dir, summary_file_path=args.summary_file)


    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the directory exists and the path is correct.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure the directory contains .fast5, .pod5, or .slow5 files.")
    except Exception as e: 
        print(f"An unexpected error occurred: {e}")


    
    


if __name__ == '__main__' : 
    main()






