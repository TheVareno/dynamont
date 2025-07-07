"""
Last script 
Use of polyA.py as package 
to compare our model performance with tailfindr, nanopolish and annotation
read the main merged read ids 
read ids on them we performed annotation, tailfindr and nanopolish 
let the model perform on the same read ids as what we have in merged table for ecoli  
merge it to initial merged-dataset
"""

import polyA_package.polyA as pa  
import pandas as pd # type: ignore
from hampel import hampel # type: ignore
from read5.Reader import read # type: ignore
import argparse
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore 


def read_merged_data(path_to_merged_data: str): 
    merged_data = pd.read_csv(path_to_merged_data)
    return merged_data


def get_read_ids(dataframe: pd.DataFrame, column='read_id'): 
    return dataframe[column]

def get_signal_values(fast5, read_id):
    """
    gets signals values of read ids which were saved in merged dataset
    filters the signal values using hampel  
    return signal values 
    """
    z_normalized_sig_val = fast5.getZNormSignal(read_id, mode='mean')
    result = hampel(z_normalized_sig_val, window_size = 5, n_sigma= 6.0) 
    filtered_z_sig_vals = result.filtered_data 
    return filtered_z_sig_vals


def save_coordinates(borders: str, path_to_file: str, read_id):
    """
    reads models predicted borders  
    polyA coordinates are first and second elements of list 
    """
    borders = borders.split(' ')
    with open(path_to_file, 'a') as coord_file: 
        coord_file.write(f'{read_id},{borders[1]}, {borders[0]}\n')  
            

def merge_model_coodinates(path_to_ann_tfr_nnps: str, path_to_model_coordinates: str): 
    
    """
    makes the final dataset
    combination of ann, tfr, nnps and model coordinates 
    """
    ann_tfr_nnps_data = read_merged_data(path_to_ann_tfr_nnps)     
    model_coords = pd.read_csv(path_to_model_coordinates)

    final_dataset = pd.merge(ann_tfr_nnps_data, model_coords, on='read_id')
    
    return final_dataset    
    

def plot_tools_error(path_to_final_dataset: str): 
    
    """
    statistical analysis of errors 
    """
    
    dataset = pd.read_csv(path_to_final_dataset) 
    
    used_tools = 'tailfindr nanopolish model'.split(' ') 
    
    for tool in used_tools: 
        dataset[f'error_{tool}_start'] = dataset[f'tail_start_{tool}'] - dataset['tail_start_annote'] 
        dataset[f'error_{tool}_end'] = dataset[f'tail_end_{tool}'] - dataset['tail_end_annote'] 
    
    dataset_long_start = pd.melt(dataset, id_vars=['read_id'], value_vars=['error_tailfindr_start', 'error_nanopolish_start', 'error_model_start'],
                  var_name='Tool', value_name='Start_Error')        
    
    dataset_long_end = pd.melt(dataset, id_vars=['read_id'], value_vars=['error_tailfindr_end', 'error_nanopolish_end', 'error_model_end'],
                      var_name='Tool', value_name='End_Error')


    avg_start_errors = dataset_long_start.groupby('Tool')['Start_Error'].mean()
    avg_end_errors = dataset_long_end.groupby('Tool')['End_Error'].mean()

    

    custom_palette = {
        'error_tailfindr_start': '#6395ee',  
        'error_nanopolish_start': '#ff991c', 
        'error_model_start': '#4cbb17',
        'error_tailfindr_end': '#6395ee',    
        'error_nanopolish_end': '#ff991c',   
        'error_model_end': '#4cbb17'         
    }

    
    
    plt.figure(figsize=(30, 20))
    plt.suptitle('Model Evaluation on E.Coli Reads Dataset', fontweight='bold', fontsize=20)
    
    
    
    # Start Errors
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Tool', y='Start_Error', data=dataset_long_start, fill=False, hue='Tool', palette=custom_palette, fliersize=11.0 )
    plt.title('Error Distribution for Start Coordinates', fontweight='bold', fontsize=16)
    plt.ylabel('Start Error', fontweight='bold', fontsize=12)
    plt.ylim(-100, 50)
    plt.axhline(0, color='#d3d3d3', linestyle='--')  # Perfect prediction line
    plt.xticks([0, 1, 2], ['Tailfindr Start', 'Nanopolish Start', 'Model Start'], fontweight='bold', fontsize=12)
    plt.xlabel('')  

    
    start_legend_labels = [
    f"{avg_start_errors['error_tailfindr_start']:.2f}",
    f"{avg_start_errors['error_nanopolish_start']:.2f}",
    f"{avg_start_errors['error_model_start']:.2f}"
    ]
    plt.legend(start_legend_labels, loc='lower right', fontsize=12, title="Average Start Errors", title_fontproperties={'weight': 'bold'})

    
    # End Errors
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Tool', y='End_Error', data=dataset_long_end, fill=False, hue='Tool', palette=custom_palette, fliersize=11.0)
    plt.title('Error Distribution for End Coordinates', fontweight='bold', fontsize=16)
    plt.ylabel('End Error', fontweight='bold', fontsize=12)
    plt.ylim(-250, 150)
    plt.axhline(0, color='#d3d3d3', linestyle='--')  # Perfect prediction line
    plt.xticks([0, 1, 2], ['Tailfindr End', 'Nanopolish End', 'Model End'], fontweight='bold', fontsize=12)
    plt.xlabel('')  
    
    end_legend_labels = [
    f"{avg_end_errors['error_tailfindr_end']:.2f}",
    f"{avg_end_errors['error_nanopolish_end']:.2f}",
    f"{avg_end_errors['error_model_end']:.2f}"
    ]
    plt.legend(end_legend_labels, loc='lower right', fontsize=12, title="Average End Errors", title_fontproperties={'weight': 'bold'})
    
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.87 , bottom=0.1)
    #plt.tight_layout(pad=6.0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.1, hspace=0.5)  
    plt.show()
    
    
    


def main(): 
    
    pa.setup_working_directory()     
    pa.set_style()
    
    parser = argparse.ArgumentParser(description="Process -s, -c and -t flags.")
    parser.add_argument('-s', action='store_true', help='Run the program to segment the signal')
    parser.add_argument('-m', action='store_true', help='Run the program to merge coordinates')
    parser.add_argument('-p', action='store_true', help='Run the program to plot the tools performance')
    args = parser.parse_args()


    merged_df = read_merged_data('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/annotation_comparison/annotations/merged_tfr_nnps_ann_readids.csv')
    read_ids = get_read_ids(merged_df) 
        
    ecoli_fast5 = pa.read_fast5('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/annotation_comparison/first_fast5/FAX28269_36c48ee6_b042d0cd_0.fast5')
    all_reads = ecoli_fast5.getReads()
    
    
    if args.s:   # segment mode   
        """
        used_read_ids = [] 
        for read_id in all_reads: 
            for wanted_read_id in read_ids: 
                if read_id == wanted_read_id: 
                    used_read_ids.append(read_id)
        
        print(f'How many reads we are going to use?\t{len(used_read_ids)}') 
        
        """
        trained_parameters = './updated_parameters.txt' 
        #untrained_parameters = './nnps_initial_parameters.txt' 

        # call segment from polyA for each read id
        
        
        used_read_ids = ['88980d1f-f5f7-4fa9-9d52-92bdb2a6645f']
        #used_read_ids = ['be881192-63da-4c8c-9ff0-e108b2be1243']
        #used_read_ids = ['e5889503-59dc-40a7-bbb7-f5ec5b0c8d38']
        #used_read_ids = ['fee4d0a2-9ed9-4539-9583-dd83e55b2c3f']
        #used_read_ids = ['26bdfaf4-8350-495a-96ab-cf7d622adcd4']
        
        counter = 0 
        for read_id in used_read_ids: 
            
            try: 
                sig_vals = get_signal_values(ecoli_fast5, read_id)
                parameters_dict = pa.get_parameters(trained_parameters)
                borders = pa.call_polyA_cpp_segment(sig_vals, parameters_dict)
                pa.plot_borders(app_output=borders, sig_values=sig_vals, read_id=read_id) 
                #pa.save_polyA_coordinates(borders=borders, read_id=read_id)
                #save_coordinates(borders=borders, path_to_file='/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building/model_RESULT/FINAL_Datasets/trained_version/model_coords_prediction.csv', 
                #                 read_id=read_id)
            
            except IndexError:  
                counter +=1 # How many non-borders ?    
                continue

        print(f'How many non-border read?\t{counter}')
            
    
    if args.m: # merged mode    
         
        model_predicted_coords = '/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building/model_RESULT/FINAL_Datasets/trained_version/model_coords_prediction.csv'

        final_dataset = merge_model_coodinates('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/annotation_comparison/annotations/merged_tfr_nnps_ann_readids.csv', model_predicted_coords)
        final_dataset.to_csv('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building/model_RESULT/FINAL_Datasets/trained_version/final_dataset_4_tools.csv')    
        
    
    if args.p: 
        
        plot_tools_error('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building/model_RESULT/FINAL_Datasets/trained_version/final_dataset_4_tools.csv') 
        



if __name__ == '__main__' : 
    main()

