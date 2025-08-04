

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
from multiprocessing import queues
import os   
import math 
import queue
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
def calculate_sample_rate(bam_file: str, mode='rb')-> dict: 

    samfile = pysam.AlignmentFile(bam_file, mode, check_sq=False) 
    read_id_length_pairs = {} 
    
    for read in samfile.fetch(until_eof=True): 
        
        read_name = read.query_name
        tags = dict(read.tags) 
        read_nt_length = int(read.query_length)
        ns = int(tags['ns']) 
        read_avg_sig_per_nt = ns / read_nt_length 

        read_id_length_pairs.update({read_name: read_avg_sig_per_nt})
    
    return read_id_length_pairs

def find_polya(task_queue: mp.Queue, result_queue: mp.Queue, input_file: str): 

    read_object = read(input_file)

    while True:
        
        try:
            read_id = task_queue.get_nowait() 
            if read_id is None: 
                break 
          
            z_normalized_signal_values = read_object.getZNormSignal(read_id, mode='mean')
            filter_object = hampel(z_normalized_signal_values, window_size=5, n_sigma=6.0)
            filtered_signal_values = filter_object.filtered_data
            
            if len(filtered_signal_values) == 0:
                print(f"[WARN] Empty filtered signal for read: {read_id}")
                continue
            
            sig_vals_str = ','.join(map(str, filtered_signal_values))
            
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
                
            else: 
                print(f"[ERROR] polyA finder exited with code {rc} for read {read_id}")
                if stderr: 
                    print(f"[STDERR] Error for {read_id}: {stderr}")
        
        except Exception as e:
            print(f"[EXCEPTION] Read {read_id} failed: {e}")
            continue
        
        result_queue.put(None)


def start_finder(input_file: str, output_path: str, sample_rates: dict):
    
    if not os.path.exists(output_path):  
        os.makedirs(output_path) 
    
    save_file = os.path.join(output_path, f'length_estimations.csv')

    with open(save_file, 'w') as f: # file exist. check 
        f.write("Read ID, poly(A) start, poly(A) end, poly(A) estimated length\n")
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    read_object = read(input_file)         
    all_read_ids = read_object.getReads() 

    for read_id in all_read_ids:
        task_queue.put(read_id)

    num_processes = 4
    for _ in range(num_processes):
        task_queue.put(None)

    processes = [mp.Process(target=find_polya, args=(task_queue, result_queue, input_file))
         for _ in range(num_processes)]
    
    for process in processes:
        process.start()

    results = [] 
    completed_results = 0 
    completed_workers = 0 
    total_tasks = len(all_read_ids)

    """
    while completed_workers < total_tasks: 
        try:                 
            read_id, borders = result_queue.get(timeout=2) 
            borders = borders.split(',') 
            pA_est_len = (int(borders[0]) - int(borders[1])) / sample_rates[read_id]
            results.append((read_id, borders[1], borders[0], math.ceil(pA_est_len)))
            print("[INFO] Completed!")
            completed += 1

        except mp.queues.Empty: 
            print("[WARN] Result queue empty for 2s - waiting...")
            continue

        except Exception as e: 
            print(f"[ERROR] Failed processing result: {e}")

    for process in processes:
        process.join()

    with open (save_file, 'a') as f: 
        for result in results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
    
    print(f"[INFO] Done. Wrote {len(results)} read estimations to {save_file}.")
    """
    while completed_workers < num_processes:
        try:                 
            result = result_queue.get(timeout=5)
            
            if result is None:  # Worker finished signal
                completed_workers += 1
                print(f"[INFO] Worker {completed_workers}/{num_processes} completed")
            else:
                read_id, borders = result
                borders = borders.split(',') 
                pA_est_len = (int(borders[0]) - int(borders[1])) / sample_rates[read_id]
                results.append((read_id, borders[1], borders[0], math.ceil(pA_est_len)))
                completed_results += 1
                print(f"[INFO] Processed result {completed_results}/{total_tasks}")

        except mp.queues.Empty:  
            print("[WARN] Result queue empty for 5s - waiting...")
            alive_processes = [p for p in processes if p.is_alive()]
            if not alive_processes:
                print("[WARN] All processes finished but still waiting for results")
                break

        except Exception as e: 
            print(f"[ERROR] Failed processing result: {e}")

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Write results to file
    with open(save_file, 'a') as f: 
        for result in results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
    
    print(f"[INFO] Done. Wrote {completed_results} read estimations to {save_file}.")
    print(f"[INFO] Processed {completed_results}/{total_tasks} tasks successfully")






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
    
    

    
    


if __name__ == '__main__' : 
    main()


