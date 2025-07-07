
"""
Author : Hadi Vareno 
Email : mohammad.noori.vareno@uni-jena.de
Github:  https://github.com/TheVareno
"""


import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from scipy import stats  # type: ignore
from read5.Reader import read # type: ignore
import ont_fast5_api.fast5_interface     # type: ignore
import random as rnd 
import os 
import sys
import csv 
from fitter import Fitter # type: ignore 
from fitter import histfit # type: ignore 
from hampel import hampel # type: ignore



def set_style(): 
    sns.set_style('darkgrid')


def setup_working_directory(): 
    if os.getcwd() != '/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building' : 
        os.chdir('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building')
    

def read_fast5(path_to_fast5: str): 
    return read(path_to_fast5)
    

def z_normalize(path_to_fast5: str, read_id: str):
    fast5 = read_fast5(path_to_fast5)
    return fast5.getZNormSignal(read_id, mode='mean') 
    

def read_csv(csv: str):
    df = pd.read_csv(csv)
    return df


def get_random_readid(df: pd.DataFrame, column='read_id'):
    return rnd.choice(df[column].values)


def save_readid(df: pd.DataFrame, txt_file: str, column='read_id'):
    read_id = get_random_readid(df, column)
    added = set()
    
    with open(txt_file, 'r') as used_readid_txt:
        for line in used_readid_txt:
            added.add(line.strip())  
    
    if read_id in added: # because the read_ids are limited 
        print('The read ID is already added.')
        return save_readid(df, txt_file, column) # find another read_id recursively
    
    else:
        with open(txt_file, 'a') as txt:
            txt.write(read_id + '\n')
            print('\n---------------------------------------')
            print(f'Read ID {read_id} added to {txt_file}')
            print('---------------------------------------\n')

            return read_id


def get_unused_read_ids(path_to_merged_file: str, path_to_used_read_ids: str, path_to_unused_read_ids: str):
    
    ann_df = pd.read_csv(path_to_merged_file) 
    
    with open(path_to_used_read_ids, 'r') as used_read_ids: 
        all_used_read_ids = [line.rstrip() for line in used_read_ids]
    
    with open (path_to_unused_read_ids, 'a') as unsed_read_ids: 
        
        for ann_read_id in ann_df['read_id']: 
            
            if ann_read_id not in all_used_read_ids: 
                unsed_read_ids.write(f'{ann_read_id}\n') 
            
        
# main functionality of this signal     
def segment_signal(path_to_polyA_annotations: str, path_to_used_read_ids_txtfile: str, path_to_fast5: str):
    
    annotations_df = pd.read_csv(path_to_polyA_annotations)  # Get the annotation info we have done on polya: 100 read_ids!
    with open('HMM/unused_read_ids.txt', 'r') as unused_read_ids: 
        
        boundaries_df = pd.DataFrame(columns='read_id start_start start_end leader_start leader_end adaptor_start adaptor_end polyA_start polyA_end transcript_start transcript_end'.split(' '))
        
        read_ids = unused_read_ids.readlines()
        
        if len(read_ids) == 1: print("this is the last unused read id ") 
                
        i = 0 
        
        for read_id in read_ids:

            read_id = read_id.strip()  # Remove any whitespace characters (including newlines)
            
            # Perform segmentation
            normalized_signals = z_normalize(path_to_fast5, read_id) 
            truncated_read_id = f"{read_id[:6]}...{read_id[-6:]}"
            fig = plt.figure(figsize=(20, 15))

            # 1. Plot the main signal 
            plt.plot(normalized_signals, color='#005073', linewidth=0.8)
            plt.title(f'Raw Signal of Read "{truncated_read_id}"', fontweight='bold')
            plt.xlabel('Signal Position')
            plt.ylabel('Normalized Signal Intensity')
            plt.xlim(0, len(normalized_signals) - 1)
            min_signal, max_signal = min(normalized_signals), max(normalized_signals)
            plt.ylim(min_signal - 0.1 * abs(min_signal), max_signal + 0.1 * abs(max_signal))  # Adjusting y-limits
            plt.pause(1)

            # 2. Mark polyA boundary    
            read_id_info = annotations_df[annotations_df['read_id'] == read_id]    
            polya_start = read_id_info['start'].values[0]
            polya_end = read_id_info['end'].values[0]
            plt.axvspan(xmin=polya_start, xmax=polya_end, color='#4CBB17', alpha=0.5, label='polyA')   
            plt.pause(1)

            # 4. Mark leader boundary
            leader_start = int(input('leader start coordinate: '))
            leader_end = int(input('leader end coordinate: '))
            plt.axvspan(xmin=leader_start, xmax=leader_end, color='#FF69B4', alpha=0.6, label='leader')
            plt.pause(1)    

            # 3. Mark start boundary 
            start_start = 0
            start_end = leader_start - 1
            plt.axvspan(xmin=start_start, xmax=start_end, color='#69b4ff', label='start', alpha=0.5)
            plt.pause(1)

            # 5. Mark adaptor boundary
            adaptor_start = leader_end + 1
            adaptor_end = polya_start - 1
            plt.axvspan(xmin=adaptor_start, xmax=adaptor_end, color='#F6BE00', alpha=0.4, label='adaptor')
            plt.pause(1)    

            # 6. Mark transcript boundary  
            transcript_start = polya_end + 1
            transcript_end = len(normalized_signals) - 1 
            plt.axvspan(xmin=transcript_start, xmax=transcript_end, color='#966FBB', alpha=0.4, label='transcript')
            plt.pause(1)    

            # Add legend for boundaries
            fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))  
            plt.savefig(f'HMM/signal_segmentation/plots/{read_id}.png', bbox_inches='tight')

            print('\n--------------------------------------------------')
            print('Saved the signal plot with boundaries successfully.')
            print('--------------------------------------------------\n')

            plt.show()

            boundaries_df.loc[len(boundaries_df.index)] = [read_id, start_start, start_end, leader_start, leader_end, adaptor_start, adaptor_end, polya_start, polya_end, transcript_start, transcript_end]

            read_ids.pop(i)  # Remove the used read_id from the list
            i = i + 1
            
            print(f'\n{len(read_ids)} unused reads to go\n')
            
            with open(path_to_used_read_ids_txtfile, 'a') as used_read_ids : 
                 used_read_ids.write(read_id + '\n')
            
            answer = str(input("annotate another read ?"))
            if answer == 'no' : break
            
        with open('HMM/unused_read_ids.txt', 'w') as unused_read_ids_file:
            unused_read_ids_file.writelines(line for line in read_ids)

        boundaries_df.to_csv('HMM/signal_segmentation/boundaries.csv', mode='a', header=False, index=False)


def filter_signal_values(path_to_used_read_ids_txtfile: str,  path_to_fast5: str):

    read_ids = []
    with open(path_to_used_read_ids_txtfile, 'r') as used_read_ids: 
        read_ids = [read_id.rstrip() for read_id in used_read_ids]
    
    
    bunch_filtered_signal_values = {}
    count = 0 
    for read_id in read_ids: 
        count = 1
        
        print(f'\nbeginning the porcess of filtering for read : {read_id}')
        
        normalized_signals = z_normalize(path_to_fast5, read_id)
        result = hampel(normalized_signals, window_size = 15, n_sigma= 14.0) 
        filtered_signal_values = result.filtered_data 
        outlier_indices = result.outlier_indices

        truncated_read_id = f"{read_id[:6]}...{read_id[-6:]}"
        
        #print(f'\nfiltering the read : {truncated_read_id} completed')

        # FILTER PLOT 
        fig, axes = plt.subplots(2, 1, figsize=(20, 15))
        fig.suptitle(f'Original Signals vs. Filtered Signals Value for Read: {truncated_read_id}', fontweight='bold', fontsize=20)
        
        fig.supxlabel('Data Points', fontweight='bold', fontsize=16)
        fig.supylabel('Signal Values', fontweight='bold', fontsize=16)

        sns.lineplot(x=range(len(normalized_signals)), y=normalized_signals, label='Original Signal', color='#7F00FF', ax=axes[0])
        #axes[0].set_title('Original Signal Before Applying Hampel Filter', fontweight='bold', fontsize=16, )

        for i in outlier_indices:
            axes[0].plot(i, normalized_signals[i], 'ro', markersize=5)  # Mark as red

        original_min, original_max = np.min(normalized_signals), np.max(normalized_signals)
        axes[0].set_ylim(([original_min - 0.05 * abs(original_min), original_max + 0.05 * abs(original_max)]))

        sns.lineplot(x=range(len(filtered_signal_values)), y=filtered_signal_values, label='Filtered Signal', color='#4cbb17', ax=axes[1])
        #axes[1].set_title('Filetered Signal Values After Applying Hampel Filter', fontweight='bold', fontsize=16, color='#2e6f40')
        axes[1].set_ylim(([original_min - 0.05 * abs(original_min), original_max + 0.05 * abs(original_max)]))
    
        #plt.savefig(f'HMM/signal_segmentation/filtered_plots/{read_id}.png')
        plt.tight_layout(pad=2.0)
        plt.show() 
        
        #print(f'the read : {truncated_read_id} was filtered successfully.\n')

        bunch_filtered_signal_values[read_id] = filtered_signal_values

        if count == 1 : 
            break
        
        
    print('Filtering Process Finished.')
    return bunch_filtered_signal_values

def fit_distributions(region: str, bunch: np.array):
    
    if bunch.size == 0:
        print(f"No valid signal values were found for region {region}.")
        return
    else :
        print(f'Here we go to fit for {region.upper()}!')
    
    # start fitting
    fitter = Fitter(bunch, 
                    distributions=['norm', 't', 'gamma', 'beta', 'expon', 'lognorm', 'uniform', 'weibull_min', 'weibull_max', 'pareto', 'chi2', 'f', 'laplace', 'gumbel_r', 'gumbel_l', 'invgauss'], 
                    timeout=300)
    fitter.fit()
    
    plt.figure(figsize=(20, 15))
    fitter.hist()
    fitter.plot_pdf(Nbest=3, lw=2, names=None, )
    plt.title(f'Distribution Fit for Region: {region.upper()}', fontsize=16, fontweight='bold')
    plt.xlabel('Signal Intensity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    plt.savefig(f'hmm/regions/{region}/plots_60_reads/distribution_fit_{region}.png', bbox_inches='tight')
    plt.show()
    
    
    # dict 
    best_fits = fitter.get_best(method='sumsquare_error')    
    
    
    with open(f'hmm/regions/{region}/fitted_distribution_{region}.txt', 'w') as fitted_info:
        
        fitted_info.write(f"Region : {region.upper()}\n")
        fitted_info.write('\n')    
        fitted_info.write("Top Best Fitting Distribution with Details :\n")
        
        for (dist_name, dist_params) in best_fits.items():
            fitted_info.write(f"\nDistribution Name: {dist_name}\n")
            fitted_info.write("   Parameters:\n")
            for param_name, param_value in dist_params.items():
                fitted_info.write(f"      {param_name}: {param_value:.6f}\n")
                    
        fitted_info.write('\n--------------------------------------------------\n')    
        fitted_info.write('Summary of 3 Best Fitted Distributions:\n')
        fitted_info.write('\n')    
        # df to string 
        summary_df = fitter.summary(Nbest=3, plot=False)
        summary_str = summary_df.to_string(header=True, index=True)
        fitted_info.write(summary_str)

    print(f'fitting done for region {region}')

        
def slice_fit(bunch_filtered_signal_values: dict, path_to_boundaries_info: str,):
    
    boundaries = pd.read_csv(path_to_boundaries_info)

    regions = 'start leader adaptor polyA transcript'.split(' ')
    
    for region in regions: 
        
        bunch_filtered_signal_values_regionwise = np.array([])
        
        print(f'Region : {region}  \n')
        print(f'\tslicing and plotting histogramms...\n')
        
        for read_id in boundaries['read_id']:    
            
            filtered_signal_values = bunch_filtered_signal_values[read_id]
             
            region_start = boundaries[boundaries['read_id'] == read_id][f'{region}_start'].values[0]
            region_end = boundaries[boundaries['read_id'] == read_id][f'{region}_end'].values[0]
    
            # Slice the signal after region 
            sliced_filtered_signal_values = filtered_signal_values[region_start:region_end]

            bunch_filtered_signal_values_regionwise = np.append(bunch_filtered_signal_values_regionwise, sliced_filtered_signal_values)
    
            truncated_read_id = f"{read_id[:6]}...{read_id[-6:]}"
            
            # Histogram  
            
            color_map = {
                'start': '#097FED',
                'polyA': '#3BB417',
                'adaptor': '#D15F00',
                'transcript': '#8e44ad', 
                'leader' : '#e75480'
            } 

            color = color_map.get(region)

            plt.figure(figsize=(20, 15))
            sns.histplot(sliced_filtered_signal_values, kde=True, bins=30, alpha=0.5, color=color)
            plt.xlabel('Z-normalized signal intensity values (Filtered via Hampel)', fontsize=12)

            plt.title(f'Distribution of Signal Intensity Values of Region {region.upper()} in Read : "{truncated_read_id}"', fontweight='bold', fontsize=16)

            plt.savefig(f'hmm/regions/{region}/plots_60_reads/histogramms/{read_id}.png', bbox_inches = 'tight')
            plt.close()    
            
            #print(f'filtering and plotting read : {truncated_read_id} done.\n')
        
        
        print(f'\tfitting distributions...\n')
        fit_distributions(region=region, bunch=bunch_filtered_signal_values_regionwise)
        print(f'Region {region} Done.\n')

        
         
def main(): 
    
    set_style()
    setup_working_directory()
    
    polyA_annotatation_csv = '../annotation_comparison/annotations/all_annotated_coordinates.csv'
    polyA_merged_csv = '../annotation_comparison/annotations/merged_tfr_nnps_ann_readids.csv'
    used_read_ids_text_file = 'HMM/used_read_ids.txt'
    unused_read_ids_text_file = 'HMM/unused_read_ids.txt'
    path_to_fast5_file = '../annotation_comparison/first_fast5/FAX28269_36c48ee6_b042d0cd_0.fast5'
    boundaries_file = 'HMM/signal_segmentation/boundaries.csv'   
    
    
   # ------------------------ SEGEMTATION & ANNOTATION ------------------------------ 
   
    #get_unused_read_ids(path_to_merged_file=polyA_merged_csv, path_to_unused_read_ids=unused_read_ids_text_file, path_to_used_read_ids=used_read_ids_text_file)
    
    #segment_signal(path_to_polyA_annotations=polyA_annotatation_csv, path_to_used_read_ids_txtfile=used_read_ids_text_file, path_to_fast5=path_to_fast5_file)
    
   # ----------------------------- FIT & FILTER ------------------------------------ 
   
    bunch_dict = filter_signal_values(path_to_used_read_ids_txtfile=used_read_ids_text_file, path_to_fast5=path_to_fast5_file)
    
   # slice_fit(bunch_filtered_signal_values=bunch_dict, path_to_boundaries_info=boundaries_file)






if __name__ == '__main__' : 
    main()


