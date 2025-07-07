
"""
"""

import polyA_package.polyA as pa # type: ignore  
import pandas as pd # type: ignore
from hampel import hampel # type: ignore
from read5.Reader import read # type: ignore
import argparse
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore 
import os

def set_directory():
    if os.getcwd() != '/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ivt' : 
        os.chdir('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ivt')
    else: 
        pass


def set_style():     
    sns.set_style('darkgrid')


def merge_datasets():
    
    # Read nnps
    nnps_df = pd.read_csv('./nanopolish/ivt_polyA_nanopolish_output.tsv', sep='\t')
    nnps_df_coords = nnps_df[['readname','polya_start', 'transcript_start']]  
    nnps_df_coords.columns = ['read_id', 'tail_start_nnps', 'tail_end_nnps']
    
    # Read tfr
    tfr_df = pd.read_csv('./tailfindr/ivt_polyA_tailfindr_output.csv')
    tfr_df_coords = tfr_df[['read_id', 'tail_start', 'tail_end']]  
    tfr_df_coords.columns = ['read_id', 'tail_start_tfr', 'tail_end_tfr']
    merged_ann_tfr = pd.merge(nnps_df_coords, tfr_df_coords, on='read_id')
    
    # read ann     
    ann_df = pd.read_csv('./annotations/coordinates.txt')
    ann_df.columns = ['read_id', 'tail_start_ann', 'tail_end_ann']
    
    # main dataframe 
    merged_nnps_tfr_ann = pd.merge(merged_ann_tfr, ann_df, on='read_id')
    return merged_nnps_tfr_ann


def read_fast5(path_to_fast5: str): 
    return read(path_to_fast5)


def save_coordinates(borders: str, path_to_file: str, read_id):
    """
    reads models predicted borders  
    polyA coordinates are first and second elements of list 
    """
    borders = borders.split(' ')
    with open(path_to_file, 'a') as coord_file: 
        coord_file.write(f'{read_id},{borders[1]}, {borders[0]}\n')  
    

def plot_tools_error(dataset: pd.DataFrame): 
    
    """
    statistical analysis of errors 
    """
    
    #dataset = pd.read_csv(path_to_final_dataset) 
    
    used_tools = 'tfr nnps mdl'.split(' ') 
    
    for tool in used_tools: 
        dataset[f'error_{tool}_start'] = dataset[f'tail_start_{tool}'] - dataset['tail_start_ann'] 
        dataset[f'error_{tool}_end'] = dataset[f'tail_end_{tool}'] - dataset['tail_end_ann'] 
    
    dataset_long_start = pd.melt(dataset, id_vars=['read_id'], value_vars=['error_tfr_start', 'error_nnps_start', 'error_mdl_start'],
                  var_name='Tool', value_name='Start_Error')        
    
    dataset_long_end = pd.melt(dataset, id_vars=['read_id'], value_vars=['error_tfr_end', 'error_nnps_end', 'error_mdl_end'],
                      var_name='Tool', value_name='End_Error')

    
    avg_start_errors = dataset_long_start.groupby('Tool')['Start_Error'].mean()
    avg_end_errors = dataset_long_end.groupby('Tool')['End_Error'].mean()


    custom_palette = {
        'error_tfr_start': '#6395ee',  
        'error_nnps_start': '#ff991c', 
        'error_mdl_start': '#4cbb17',
        'error_tfr_end': '#6395ee',    
        'error_nnps_end': '#ff991c',   
        'error_mdl_end': '#4cbb17'         
    }
    
    plt.figure(figsize=(30, 20))
    plt.suptitle('Model Evaluation on IVT Reads Dataset', fontweight='bold', fontsize=20)
    

    # Start Errors
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Tool', y='Start_Error', data=dataset_long_start, fill=False, hue='Tool', palette=custom_palette, fliersize=11.0)
    plt.title('Error Distribution for Start Coordinates', fontweight='bold', fontsize=16)
    plt.ylabel('Start Error', fontweight='bold', fontsize=12)
    plt.ylim(-100, 200)
    plt.axhline(0, color='#d3d3d3', linestyle='--')  
    plt.xticks([0, 1, 2], ['Tailfindr Start', 'Nanopolish Start', 'Model Start'], fontweight='bold', fontsize=12)
    plt.xlabel('')  

    
    start_legend_labels = [
    f"{avg_start_errors['error_tfr_start']:.2f}",
    f"{avg_start_errors['error_nnps_start']:.2f}",
    f"{avg_start_errors['error_mdl_start']:.2f}"
    ]
    plt.legend(start_legend_labels, loc='lower right', fontsize=12, title="Average Start Errors", title_fontproperties={'weight': 'bold'})

    
    
    # End Errors
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Tool', y='End_Error', data=dataset_long_end, fill=False, hue='Tool', fliersize=11.0)
    plt.title('Error Distribution for End Coordinates', fontweight='bold', fontsize=16)
    plt.ylabel('End Error', fontweight='bold', fontsize=12)
    plt.ylim(-350, 300)
    plt.axhline(0, color='#d3d3d3', linestyle='--')  
    plt.xticks([0, 1, 2], ['Tailfindr End', 'Nanopolish End', 'Model End'], fontweight='bold', fontsize=12)
    plt.xlabel('')  
    
    start_legend_labels = [
    f"{avg_end_errors['error_tfr_end']:.2f}",
    f"{avg_end_errors['error_nnps_end']:.2f}",
    f"{avg_end_errors['error_mdl_end']:.2f}"
    ]
    plt.legend(start_legend_labels, loc='lower right', fontsize=12, title="Average End Errors", title_fontproperties={'weight': 'bold'})

    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.1, hspace=0.5)  

    plt.show()
      


def main(): 
    
    set_directory()
    set_style() 
    
    # main dataframe 
    merged_nnps_tfr_ann = merge_datasets()

    # extract read ids for model evaluation 
    ivt_reads = merged_nnps_tfr_ann['read_id']
    
    # ivt fast5
    ivt_fast5 = read_fast5('./FAO07649_853a2ebb_0.fast5')
    
    trained_parameters = '../../scripts/model_building/polyA_model/updated_parameters.txt' 
    #untrained_parameters = '../../scripts/model_building/polyA_model/nnps_initial_parameters.txt'
    
    parser = argparse.ArgumentParser(description="Process -s, -c and -t flags.")
    parser.add_argument('-s', action='store_true', help='Run the program to segment the signal')
    parser.add_argument('-m', action='store_true', help='Run the program to merge coordinates')
    parser.add_argument('-p', action='store_true', help='Run the program to plot the tools performance')
    args = parser.parse_args()

    if args.s : 
    
        counter = 0
        for read in ivt_reads: 
            try:
                z_normalized_sig_val = ivt_fast5.getZNormSignal(read, mode='mean')
                result = hampel(z_normalized_sig_val, window_size = 5, n_sigma= 6.0)  
                filtered_z_sig_vals = result.filtered_data

                parameters_dict = pa.get_parameters(trained_parameters)

                borders = pa.call_polyA_cpp_segment(filtered_z_sig_vals, parameters_dict)

                pa.plot_borders(app_output=borders, sig_values=filtered_z_sig_vals, read_id=read) 

                #pa.save_polyA_coordinates(borders=borders, read_id=read)

                save_coordinates(borders=borders, path_to_file='./model_RESULT/FINAL_Datasets/trained_version/model_coords_prediction.txt', read_id=read)

            except: 
                counter += 1
                continue
        
        print(f'How many non-border read?\t{counter}')

    if args.m: 
       
        model_predicted_coords = './model_RESULT/FINAL_Datasets/trained_version/model_coords_prediction.txt'
        
        model_predicted_coords = pd.read_csv(model_predicted_coords)
        
        model_predicted_coords.columns = ['read_id', 'tail_start_mdl', 'tail_end_mdl']
        
        final_dataset = pd.merge(merged_nnps_tfr_ann, model_predicted_coords, on='read_id')
        
        final_dataset.to_csv('./model_RESULT/FINAL_Datasets/trained_version/final_dataset_4_tools.txt')    
        
    if args.p: 
        
        final_dataset_ivt = pd.read_csv('./model_RESULT/FINAL_Datasets/trained_version/final_dataset_4_tools.txt')
        plot_tools_error(final_dataset_ivt)
        
    

if __name__ == '__main__': 
    main() 

