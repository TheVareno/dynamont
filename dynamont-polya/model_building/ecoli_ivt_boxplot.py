
import pandas as pd # type: ignore 
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

def plot_tools_error(ecoli_dataset: pd.DataFrame, ivt_dataset: pd.DataFrame): 
    
    """
    Builds box plots of start and end errors for each tool on both the E.coli and IVT datasets.
    """

    # Define the tools
    used_tools = ['tailfindr', 'nanopolish', 'model']
    
    # Calculate error columns for both datasets
    for tool in used_tools: 
        for dataset in [ecoli_dataset, ivt_dataset]:
            dataset[f'error_{tool}_start'] = dataset[f'tail_start_{tool}'] - dataset['tail_start_ann']
            dataset[f'error_{tool}_end'] = dataset[f'tail_end_{tool}'] - dataset['tail_end_ann']
    
    # Melt datasets for easy plotting
    ecoli_long = pd.melt(ecoli_dataset, id_vars=['read_id'], 
                         value_vars=[f'error_{tool}_start' for tool in used_tools] + 
                                    [f'error_{tool}_end' for tool in used_tools],
                         var_name='Tool', value_name='Error')
    
    ivt_long = pd.melt(ivt_dataset, id_vars=['read_id'], 
                       value_vars=[f'error_{tool}_start' for tool in used_tools] + 
                                  [f'error_{tool}_end' for tool in used_tools],
                       var_name='Tool', value_name='Error')
    
    # Define a custom color palette
    custom_palette = {
        'error_tailfindr_start': '#6395ee', 'error_tailfindr_end': '#6395ee',
        'error_nanopolish_start': '#ff991c', 'error_nanopolish_end': '#ff991c',
        'error_model_start': '#4cbb17', 'error_model_end': '#4cbb17'
    }
    
    # Plotting
    plt.figure(figsize=(20, 10))
    plt.suptitle('Tool Error Analysis for E. coli and IVT Datasets', fontweight='bold', fontsize=20)
    
    # E.coli Dataset
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Tool', y='Error', data=ecoli_long, hue='Tool', palette=custom_palette, fliersize=8)
    plt.title('Error Distribution on E. coli Dataset', fontweight='bold', fontsize=16)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks([0, 1, 2], ['Tailfindr Start', 'Nanopolish Start', 'Model Start', 'Tailfindr End', 'Nanopolish End', 'Model End'], fontweight='bold')
    plt.xlabel('Tools', fontweight='bold')
    plt.ylabel('Error', fontweight='bold')
    
    # IVT Dataset
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Tool', y='Error', data=ivt_long, hue='Tool', palette=custom_palette, fliersize=8)
    plt.title('Error Distribution on IVT Dataset', fontweight='bold', fontsize=16)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks([0, 1, 2], ['Tailfindr Start', 'Nanopolish Start', 'Model Start', 'Tailfindr End', 'Nanopolish End', 'Model End'], fontweight='bold')
    plt.xlabel('Tools', fontweight='bold')
    plt.ylabel('Error', fontweight='bold')

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1, wspace=0.4)
    plt.show()


def main(): 
    
    ecoli_ds = '/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/model_building/model_RESULT/FINAL_Datasets/trained_version/final_dataset_4_tools.csv'
    
    ivt_ds = pd.read_csv('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ivt/model_RESULT/FINAL_Datasets/trained_version/final_dataset_4_tools.txt')
    
    plot_tools_error(ecoli_ds, ivt_ds)
    



if __name__ == '__main__':
    main()