"""
Author : Hadi Vareno 
Email : mohammad.noori.vareno@uni-jena.de
Github:  https://github.com/TheVareno
"""

import numpy as np   # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
sns.set_style("darkgrid")
from scipy import stats # type: ignore
import os
import sys, argparse  


def load_coordination_dataset(path:str, column='read_id')-> pd.DataFrame:
    file_name, file_extension = os.path.splitext(path)
        
    if file_extension == '.tsv': 
        df = pd.read_csv(path, sep='\t')
        column = 'readname'
    elif file_extension == '.csv': 
        df = pd.read_csv(path)
    else: 
        print('invalid text File')
        
    df_sorted = df.sort_values(by=[column])
    
    return df_sorted


"""
- sets annotations as grand truth, then calculates the variation of each predicted coordination. 
- plots the histograms of variations. 
"""
def plot_variation_distribution(df:pd.DataFrame, case: str)-> None: 
    if case == 'start': 
        dist_ann_tfr = df['tail_start_tailfindr'] - df['tail_start_annote']
        mean_dist_ann_tfr = dist_ann_tfr.mean()
        median_dist_ann_tfr = dist_ann_tfr.median() 
        std_dist_ann_tfr = dist_ann_tfr.std()
        skew_dist_ann_tfr = dist_ann_tfr.skew()

        dist_ann_nnps = df['tail_start_nanopolish'] - df['tail_start_annote']
        mean_dist_ann_nnps = dist_ann_nnps.mean()
        median_dist_ann_nnps = dist_ann_nnps.median()
        std_dist_ann_nnps = dist_ann_nnps.std()
        skew_dist_ann_nnps = dist_ann_nnps.skew()

        dist_tfr_nnps = df['tail_start_nanopolish'] - df['tail_start_tailfindr']
        mean_dist_tfr_nnps = dist_tfr_nnps.mean()
        median_dist_tfr_nnps = dist_tfr_nnps.median()
        std_dist_tfr_nnps = dist_tfr_nnps.std()
        skew_dist_tfr_nnps = dist_tfr_nnps.skew() 
    
    elif case == 'end': 
        dist_ann_tfr = df['tail_end_tailfindr'] - df['tail_end_annote']
        mean_dist_ann_tfr = dist_ann_tfr.mean()
        median_dist_ann_tfr = dist_ann_tfr.median() 
        std_dist_ann_tfr = dist_ann_tfr.std()
        skew_dist_ann_tfr = dist_ann_tfr.skew()

        dist_ann_nnps = df['tail_end_nanopolish'] - df['tail_end_annote']
        mean_dist_ann_nnps = dist_ann_nnps.mean()
        median_dist_ann_nnps = dist_ann_nnps.median()
        std_dist_ann_nnps = dist_ann_nnps.std()
        skew_dist_ann_nnps = dist_ann_nnps.skew()

        dist_tfr_nnps = df['tail_end_nanopolish'] - df['tail_end_tailfindr']
        mean_dist_tfr_nnps = dist_tfr_nnps.mean()
        median_dist_tfr_nnps = dist_tfr_nnps.median()
        std_dist_tfr_nnps = dist_tfr_nnps.std()
        skew_dist_tfr_nnps = dist_tfr_nnps.skew()
    
    else: 
        raise ValueError 

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    if case == 'start': fig.suptitle('Distribution of Distances in Predicted and Annotated Start Coordinates', fontsize=16, fontweight='bold')
    else: fig.suptitle('Distribution of Distances in Predicted and Annotated End Coordinates', fontsize=16, fontweight='bold')

    sns.histplot(dist_ann_tfr, kde=True, bins=30, color='#3498db', alpha=0.4, ax=axes[0])
    axes[0].axvline(mean_dist_ann_tfr, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_ann_tfr:.2f}') 
    axes[0].axvline(median_dist_ann_tfr, color='green', linestyle='--', label=f'median: {median_dist_ann_tfr:.2f}')
    axes[0].axvspan(xmin=mean_dist_ann_tfr - std_dist_ann_tfr, xmax=mean_dist_ann_tfr + std_dist_ann_tfr, color='#BCC6CC', alpha=0.3, label=f'mean ± Std: {std_dist_ann_tfr:.2f}')
    axes[0].annotate(f'skewness: {skew_dist_ann_tfr:.2f}', xy=(-5000, 50),  bbox=dict(boxstyle="round", color='#3498db' ,alpha=0.1), size = 10)
    axes[0].set_ylabel('Tailfindr vs. Annotation', fontsize=12, fontweight='bold')
    
    sns.histplot(dist_ann_nnps, kde=True, bins=30, color='#9b59b6', alpha=0.5, ax=axes[1])
    axes[1].axvline(mean_dist_ann_nnps, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_ann_nnps:.2f}') 
    axes[1].axvline(median_dist_ann_nnps, color='green', linestyle='--', label=f'median: {median_dist_ann_nnps:.2f}')
    axes[1].axvspan(xmin=mean_dist_ann_nnps - std_dist_ann_nnps, xmax=mean_dist_ann_nnps + std_dist_ann_nnps, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_ann_nnps:.2f}')
    axes[1].annotate(f'skewness: {skew_dist_ann_nnps:.2f}', xy=(-13000, 50),  bbox=dict(boxstyle="round", color='#9b59b6' ,alpha=0.1), size = 10)
    axes[1].legend(loc='upper left')
    axes[1].set_ylabel('Nanopolish vs. Annotation', fontsize=12, fontweight='bold')
    
    sns.histplot(dist_tfr_nnps, kde=True, bins=30, color='#f39c12', alpha=0.4, ax=axes[2])
    axes[2].axvline(mean_dist_tfr_nnps, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_tfr_nnps:.2f}') 
    axes[2].axvline(median_dist_tfr_nnps, color='green', linestyle='--', label=f'median: {median_dist_tfr_nnps:.2f}')
    axes[2].axvspan(xmin=mean_dist_tfr_nnps - std_dist_tfr_nnps, xmax=mean_dist_tfr_nnps + std_dist_tfr_nnps, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_tfr_nnps:.2f}')
    axes[2].annotate(f'skewness: {skew_dist_tfr_nnps:.2f}', xy=(-13000, 50),  bbox=dict(boxstyle="round", color='#f39c12' ,alpha=0.1), size = 10)
    axes[2].set_ylabel('Nanopolish vs. Tailfindr', fontsize=12, fontweight='bold')
    
    for ax in axes:
        ax.legend(loc='upper left')
        ax.set_title('')  
        ax.set_yticklabels([])  
        ax.set_xlabel('')  

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.supxlabel('Distances', fontsize=12, fontweight='bold')
    plt.show()



def plot_performance(df: pd.DataFrame, plot: str)-> None:
    df['distance_start_tfr_ann'] = df['tail_start_tailfindr'] - df['tail_start_annote'] 
    df['distance_start_nnps_ann'] = df['tail_start_nanopolish'] - df['tail_start_annote'] 
    df['distance_end_tfr_ann'] = df['tail_end_tailfindr'] - df['tail_end_annote'] 
    df['distance_end_nnps_ann'] = df['tail_end_nanopolish'] - df['tail_end_annote'] 

    if plot == 'box' or plot == 'violin':    
        melted_df = pd.melt(df, id_vars=['tail_start_annote', 'tail_end_annote'], 
                                value_vars=['distance_start_tfr_ann', 'distance_start_nnps_ann',
                                    'distance_end_tfr_ann', 'distance_end_nnps_ann'],
                                var_name='Tool_Metric', 
                                value_name='Distance')

        melted_df['Metric'] = melted_df['Tool_Metric'].apply(lambda x: 'Start' if 'start' in x else 'End')  

        melted_df['Tool'] = melted_df['Tool_Metric'].apply(lambda x: 'Tailfindr' if 'tfr' in x else 'Nanopolish')

        plt.figure(figsize=(20, 15))
        if plot == 'box': 
            sns.boxplot(x='Metric', y='Distance', hue='Tool', data=melted_df, palette={'Tailfindr': '#3498db', 'Nanopolish': '#ffa500'}, fill=False, gap=0.1, saturation=0.75, width=0.5)
        elif plot == 'violin': 
            sns.violinplot(x='Metric', y='Distance', hue='Tool', data=melted_df, palette={'Tailfindr': '#3498db', 'Nanopolish': '#f39c12'}, split=True, inner="box", scale="width", fill=False)
        else: raise ValueError
        plt.title('Distribution of Distances for Start, End, and Length by Tool', fontsize=16, fontweight='bold')
        plt.xlabel('Metric (Start, End, Length)', fontsize=12, fontweight='bold')
        plt.ylabel('Distance = Tool - Annotation', fontsize=12, fontweight='bold')
        plt.legend(title='Tool', loc='upper right',  bbox_to_anchor=(1, 1))
        plt.show()

    elif plot == 'scatter':
        mean_dist_start_tfr_ann = df['distance_start_tfr_ann'].mean()
        mean_dist_start_nnps_ann = df['distance_start_nnps_ann'].mean()
        mean_dist_end_tfr_ann = df['distance_end_tfr_ann'].mean()
        mean_dist_end_nnps_ann = df['distance_end_nnps_ann'].mean()
        
        melted_df_start = df.melt(id_vars='tail_start_annote', value_vars=['distance_start_tfr_ann', 'distance_start_nnps_ann'], var_name='Tool_Start', value_name='Distance_Start')
        melted_df_end = df.melt(id_vars='tail_end_annote', value_vars=['distance_end_tfr_ann', 'distance_end_nnps_ann'], var_name='Tool_End', value_name='Distance_End')
        palette = {'distance_start_tfr_ann': '#3498db', 'distance_start_nnps_ann': '#f39c12'} 
        palette = {'distance_end_tfr_ann': '#3498db', 'distance_end_nnps_ann': '#f39c12'} 
    
        fig, axes = plt.subplots(1, 2, figsize=(20, 15))
        sns.scatterplot(data=melted_df_start, x='tail_start_annote', y='Distance_Start', hue='Tool_Start', palette=palette, ax=axes[0], 
                   legend=False, markers={'distance_start_tfr_ann' : 'X', 'distance_start_nnps_ann' : 'o'}, s=95,  alpha=0.5)

        axes[0].axhline(mean_dist_start_tfr_ann, color='#ff2c2c', linestyle='--', label=f'Tailfindr: {mean_dist_start_tfr_ann:.2f}')
        axes[0].axhline(mean_dist_start_nnps_ann, color='#4cbb17', linestyle='--', label=f'Nanopolish: {mean_dist_start_nnps_ann:.2f}')
    
        sns.scatterplot(data=melted_df_end, x='tail_end_annote', y='Distance_End', hue='Tool_End', palette=palette, ax=axes[1], 
                       legend=False, markers={'distance_end_tfr_ann' : 'X', 'distance_end_nnps_ann' : 'o'}, s=95,  alpha=0.5)
        axes[1].axhline(mean_dist_end_tfr_ann, color='#ff2c2c', linestyle='--', label=f'Tailfindr: {mean_dist_end_tfr_ann:.2f}')
        axes[1].axhline(mean_dist_end_nnps_ann, color='#4cbb17', linestyle='--', label=f'Nanopolish: {mean_dist_end_nnps_ann:.2f}')

        for ax in axes: 
            ax.set_ylim(-100, 100)
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout()
        plt.savefig('./comparisons_plots/main_scatter.png')
        plt.show()


def save_dataframe(df:pd.DataFrame, path: str)-> None: 
    file_path = path + '/' + 'main_comparison_table.tsv'
    df.to_csv(file_path, sep='\t', encoding='utf-8', index=False, header=True)
    print('successfully saved the dataframe.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tools', choices=['tfr_nnps', 'tfr_ann', 'nnps_ann'], type=str, help='plot the performance of each tools')
    parser.add_argument('--case', choices=['variation', 'performance'], type=str, help='which cooridination')
    parser.add_argument('--coord', choices=['start', 'end'], type=str, help='which cooridination')
    parser.add_argument('--plot', choices=['box', 'violin', 'scatter'], type=str, help='how to plot')
    parser.add_argument('--path', type=str, help='the path to save')
    args = parser.parse_args()

    # load
    ann_coord = load_coordination_dataset('annotations/all_annotated_coordinates.csv')
    ann_coord.columns = ['read_id' ,'tail_start_annote', 'tail_end_annote', 'alt_tail_start', 'alt_tail_end'] # reset the col names 
    ann_coord['tail_length_annote'] = ann_coord['tail_end_annote'] - ann_coord['tail_start_annote'] # add tail_column based on tail_start and tail_end
    ann_coord.drop(['alt_tail_start', 'alt_tail_end'], axis=1, inplace=True)   

    tailfindr_coord = load_coordination_dataset('tailfindr_polya/tailfindr_after_basecalling/new_tailfindr/ecoli_164_subset_tails.csv')
    tailfindr_coord.columns = ['read_id', 'tail_start_tailfindr', 'tail_end_tailfindr', 'samples_per_nt','file_path']
    tailfindr_coord.drop(['samples_per_nt', 'file_path'], axis=1, inplace=True)
    
    nnps_coord = load_coordination_dataset('nanopolish_polya/ecoli_polya_all_reads.tsv')
    nnps_coord.columns = ['read_id', 'contig', 'position', 'leader_start', 'adapter_start',
                           'tail_start_nanopolish', 'tail_end_nanopolish', 'read_rate', 'qc_tag_nanopolish']
    nnps_coord.drop(['read_rate', 'adapter_start', 'leader_start', 'position', 'contig'], axis=1, inplace=True)

    if args.case == 'variation':
        if args.tools == 'tfr_ann':
            paired_tools = pd.merge(ann_coord, tailfindr_coord, on='read_id') # 163 rows 

        if args.tools == 'nnps_ann':
            paired_tools = pd.merge(ann_coord, nnps_coord, on='read_id') # 100 rows 

        if args.tools == 'nnps_ann':
            paired_tools = pd.merge(tailfindr_coord, nnps_coord, on='read_id') # 95 rows
    
        plot_variation_distribution(paired_tools, args.coord)
        
    if args.case == 'performance':    
        paired_tools = pd.merge(ann_coord, tailfindr_coord, on='read_id') # 163 rows 
        all_tools = pd.merge(paired_tools, nnps_coord, on='read_id')  
        all_tools = all_tools.drop_duplicates(subset=['read_id'], keep='first')

        if args.plot == 'box': 
            plot_performance(all_tools, 'box')
        
        elif args.plot == 'violin': 
            plot_performance(all_tools, 'violin')
        
        elif args.plot == 'scatter': 
            plot_performance(all_tools, 'scatter')

    if args.path: 
        save_dataframe(all_tools, args.path)
        
    

    

if __name__ == "__main__":
    main()


