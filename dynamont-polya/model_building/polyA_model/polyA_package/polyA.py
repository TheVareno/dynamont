
"""
author: Hadi Vareno
e-mail: mohammad.noori.vareno@uni-jena.de
github: https://github.com/TheVareno
"""

import numpy as np 
from read5.Reader import read # type: ignore 
import ont_fast5_api.fast5_interface     # type: ignore 
import argparse
from hampel import hampel # type: ignore
import subprocess as sp
import os  

def setup_working_directory():     
    if os.getcwd() != '/home/hi68ren/Dokumente/ProjektModul/Implementation/scripts/dynamont_polya' : 
        os.chdir('/home/hi68ren/Dokumente/ProjektModul/Implementation/scripts/dynamont_polya')
    else : 
        pass

    
def find_polya(sig_vals, read_id, save_file):
    try:
        polyA_app_call = './polyA'
        sig_vals_str = ','.join(map(str, sig_vals))
        
        process = sp.Popen(polyA_app_call, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        process.stdin.write(f"{sig_vals_str}\n{read_id}\n{save_file}")
        
        process.stdin.flush()
        stdout, stderr = process.communicate()
        
        if stderr:
            print(f"Error for read kire khar {read_id}: {stderr}")
            return read_id, None

    except Exception as e:
        print(f"Error for read kose gav {read_id}: {e}")
        return read_id, None


def process_read(read_id, read_object, save_file):
    
    z_normalized_signal_values = read_object.getZNormSignal(read_id, mode='mean')
    filter_object = hampel(z_normalized_signal_values, window_size=5, n_sigma=6.0)
    filtered_signal_values = filter_object.filtered_data
    return find_polya(filtered_signal_values, read_id, save_file)


def main():
    setup_working_directory()

    parser = argparse.ArgumentParser(description="Process and Save output file.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to ONT read data in FAST5, POD5, or SLOW5.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save output file.")
    args = parser.parse_args()

    file_path = args.input_file
    save_file = os.path.join(args.output_path, 'polya_borders.csv')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found.")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(save_file, 'w') as f:
        f.write("Read ID,Poly(A) end,Poly(A) start\n")

    read_object = read(file_path)   
    all_read_ids = read_object.getReads()

    for read_id in all_read_ids:
        process_read(read_id, read_object, save_file)


if __name__ == '__main__':
    main()



