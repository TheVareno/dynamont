
"""
Author : Hadi Vareno 
Email : mohammad.noori.vareno@uni-jena.de
Github:  https://github.com/TheVareno
"""

"""
This scirpt:  
- extracts the signal values belonging to random read id, 
- normalize the sginal values 
- plot the time series data 
- record the annotated cooridinations of poylyA on time serie data 
- reusable for R10 data annotation 

NOTE : get nanopolish estimated read_ids first! since nnps performs mapping in first hand 
then polyA length estimation, so many already annotated read ids are filtered out by nanopolish probably 
based on phred score threshold!  

"""


import pandas as pd # type: ignore
from read5.Reader import read # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns # type: ignore
sns.set_style('darkgrid')

import ont_fast5_api.fast5_interface  # type: ignore
import random as rnd
from csv import writer
import argparse
import os 


def make_read_obj(read_path: str):
    return read(read_path) 


def get_random_read(read_path: str):
    read_obj = make_read_obj(read_path)
    all_reads = read_obj.getReads()
    return rnd.choice(all_reads)


# nanopolish data
def load_nanopolish_output():
    nnps_polya_df = pd.read_csv("nanopolish_polya/ecoli_polya_all_reads.tsv", sep='\t')
    with open("annotations/used_readids.txt") as readids:
        used_read_ids = [line.rstrip() for line in readids]
    return nnps_polya_df, used_read_ids


def categorize_read_ids(nnps_read_ids, used_read_ids):
    cmn_read_ids = [read_id for read_id in nnps_read_ids if read_id in used_read_ids]
    nnps_uniq_read_ids = [read_id for read_id in nnps_read_ids if read_id not in used_read_ids]    
    print(f"Common read IDs: {len(cmn_read_ids)}")
    print(f"Unique Nanopolish read IDs: {len(nnps_uniq_read_ids)}")
    return cmn_read_ids, nnps_uniq_read_ids


def get_random_read_nnps(nnps_uniq_read_ids):
    return rnd.choice(nnps_uniq_read_ids)


def get_random_read(ecoli_ivv):
    all_reads = ecoli_ivv.getReads()
    return rnd.choice(all_reads)


# just in case
from ont_fast5_api.fast5_interface import get_fast5_file # type: ignore
def print_all_raw_data(path: str):
    with get_fast5_file(path, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            print(read.read_id, raw_data)


# selective sampling of reads - not overall usefull (ivt case)
def get_random_read_ids(input_tsv: str, output_file: str, n: int = 100):
    
    # Read the Nanopolish TSV file into a DataFrame
    df = pd.read_csv(input_tsv, sep='\t')

    sampled_read_ids = df['readname'].sample(n=n, random_state=42)

    with open(output_file, 'w') as file:
        for read_id in sampled_read_ids:
            file.write(f"{read_id}\n")


def process_read_id(input_file: str, used_file: str):
    
    if os.stat(input_file).st_size == 0:
        print("No read IDs left to process.")
        return None

    # Read remaining read_ids 
    with open(input_file, 'r') as file:
        read_ids = file.readlines()

    current_read_id = read_ids[0].strip()

    with open(used_file, 'a') as used_file_obj:
        used_file_obj.write(f"{current_read_id}\n")

    # Remove the processed read_id from the input file
    with open(input_file, 'w') as file:
        file.writelines(read_ids[1:])  # Write back the remaining IDs

    return current_read_id


# append to table
def add_coordinate(read_id: str, start: int, end: int, alt_start: any, alt_end: any)-> None:
    alt_start = None if alt_start == 'n' else alt_start
    alt_end = None if alt_end == 'n' else alt_end

    new_read_cooridinates = [read_id, start, end, alt_start, alt_end]
    new_row_str = ','.join(str(el) for el in new_read_cooridinates)
    
    with open('annotations/coordinates_records.txt', 'a', newline='') as coord_file:
        #writer_obj = writer(csv)
        #writer_obj.writerow(new_read_cooridinates)
        coord_file.write(f'{new_row_str}\n')


# add to used read ids 
def add_used_readid(read_id:str)-> None:
    with open('annotations/used_readids.txt', 'a') as txt:
        txt.write(read_id + '\n') 


def record_coordinations()-> list:
    polyA_start = int(input('polyA start coordinate: '))
    polyA_alt_start = int(input('poly alternative start coordinate: '))

    polyA_end = int(input('polyA end coordinate: '))
    polyA_alt_end = int(input('polya alternative end coordinate:'))

    return [polyA_start, polyA_end, polyA_alt_start, polyA_alt_end] 

    
def plot_signal(read_signal, read_id): 
    # show first plot     
    plt.figure(figsize=(20, 8))
    plt.plot(read_signal, color='green', linewidth=0.8)
    plt.title(f'Raw Signal of Values of the Read ID: {read_id}')
    plt.xlabel('Timestamps (miliseconds)')
    plt.ylabel('Signal Intensity (picoAmpere)')
    plt.grid(True)
    plt.show()

    # next record boundaries
    boundaries = record_coordinations()

    # append to csv file as new row
    add_coordinate(read_id=read_id, start=boundaries[0], end=boundaries[1], alt_start=boundaries[2], alt_end=boundaries[3])
    add_used_readid(read_id=read_id)

    # second plot with bounderies 
    plt.figure(figsize=(20, 8))
    plt.plot(read_signal, color='green', linewidth=0.8)

    for i in [boundaries[0], boundaries[1]]:
        plt.axvline(x = i, color='red', ls='--', lw=0.8)

    plt.title(f"Raw Signal Values of the Read ID: {read_id}")
    plt.xlabel('Timestamps (miliseconds)')
    plt.ylabel('Signal Intensity (picoAmpere)')
    plt.grid(True)
    plt.savefig(f"plots/{read_id}.png", bbox_inches = 'tight')
    plt.show()


def main():

    parser = argparse.ArgumentParser() 
    parser.add_argument('-i', '--read_file', help='path to ONT read data')
    args = parser.parse_args()


    if args.read_file:

        """
        nnps_polya_df, used_read_ids = load_nanopolish_output()
        nnps_read_ids = nnps_polya_df['readname']
        cmn_read_ids, nnps_uniq_read_ids = categorize_read_ids(nnps_read_ids, used_read_ids)
        """

        #! read id random [1, 3999)
        read_id = get_random_read(args.read_file)
        print(f'The current chosen read id : {read_id}')

        read_obj = make_read_obj(str(args.read_file))    

        #! main signal  
        read_signal = read_obj.getZNormSignal(read_id)
        plot_signal(read_signal, read_id)
    
    

if __name__ == '__main__': 
    main()