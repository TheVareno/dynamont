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


def set_working_directory():
    target_dir = '/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/annotation_comparison'
    if os.getcwd != target_dir : 
        os.chdir(target_dir)

    return target_dir



def load_sort_csv(path:str, column='read_id'):
    
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



def plot_start_ditances(df:pd.DataFrame): 

    dist_ann_tfr = df['tail_start_tailfindr'] - df['tail_start_annote']
    mean_dist_ann_tfr = dist_ann_tfr.mean()
    median_dist_ann_tfr = dist_ann_tfr.median() 
    std_dist_ann_tfr = dist_ann_tfr.std()
    var_dist_ann_tfr = dist_ann_tfr.var()
    skew_dist_ann_tfr = dist_ann_tfr.skew()

    dist_ann_nnps = df['tail_start_nanopolish'] - df['tail_start_annote']
    mean_dist_ann_nnps = dist_ann_nnps.mean()
    median_dist_ann_nnps = dist_ann_nnps.median()
    std_dist_ann_nnps = dist_ann_nnps.std()
    var_dist_ann_nnps = dist_ann_nnps.var()
    skew_dist_ann_nnps = dist_ann_nnps.skew()


    dist_tfr_nnps = df['tail_start_nanopolish'] - df['tail_start_tailfindr']
    mean_dist_tfr_nnps = dist_tfr_nnps.mean()
    median_dist_tfr_nnps = dist_tfr_nnps.median()
    std_dist_tfr_nnps = dist_tfr_nnps.std()
    var_dist_tfr_nnps = dist_tfr_nnps.var()
    skew_dist_tfr_nnps = dist_tfr_nnps.skew()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Distribution of Distances in Predicted and Annotated Start Coordinates', fontsize=16, fontweight='bold')

    sns.histplot(dist_ann_tfr, kde=True, bins=30, color='#3498db', alpha=0.4, ax=axes[0])
    axes[0].axvline(mean_dist_ann_tfr, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_ann_tfr:.2f}') 
    axes[0].axvline(median_dist_ann_tfr, color='green', linestyle='--', label=f'median: {median_dist_ann_tfr:.2f}')
    axes[0].axvspan(xmin=mean_dist_ann_tfr - std_dist_ann_tfr, xmax=mean_dist_ann_tfr + std_dist_ann_tfr, color='#BCC6CC', alpha=0.3, label=f'mean ± Std: {std_dist_ann_tfr:.2f}')
    axes[0].annotate(f'skewness: {skew_dist_ann_tfr:.2f}', xy=(-5000, 50),  bbox=dict(boxstyle="round", color='#3498db' ,alpha=0.1), size = 10)
    axes[0].legend(loc='upper left')
    axes[0].set_ylabel('Tailfindr vs. Annotation', fontsize=12, fontweight='bold')
    axes[0].set_title('')  
    axes[0].set_yticklabels([])  
    axes[0].set_xlabel('')  

    sns.histplot(dist_ann_nnps, kde=True, bins=30, color='#9b59b6', alpha=0.5, ax=axes[1])
    axes[1].axvline(mean_dist_ann_nnps, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_ann_nnps:.2f}') 
    axes[1].axvline(median_dist_ann_nnps, color='green', linestyle='--', label=f'median: {median_dist_ann_nnps:.2f}')
    axes[1].axvspan(xmin=mean_dist_ann_nnps - std_dist_ann_nnps, xmax=mean_dist_ann_nnps + std_dist_ann_nnps, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_ann_nnps:.2f}')
    axes[1].annotate(f'skewness: {skew_dist_ann_nnps:.2f}', xy=(-13000, 50),  bbox=dict(boxstyle="round", color='#9b59b6' ,alpha=0.1), size = 10)
    axes[1].legend(loc='upper left')
    axes[1].set_ylabel('Nanopolish vs. Annotation', fontsize=12, fontweight='bold')
    axes[1].set_title('')  
    axes[1].set_yticklabels([])  
    axes[1].set_xlabel('')  

    sns.histplot(dist_tfr_nnps, kde=True, bins=30, color='#f39c12', alpha=0.4, ax=axes[2])
    axes[2].axvline(mean_dist_tfr_nnps, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_tfr_nnps:.2f}') 
    axes[2].axvline(median_dist_tfr_nnps, color='green', linestyle='--', label=f'median: {median_dist_tfr_nnps:.2f}')
    axes[2].axvspan(xmin=mean_dist_tfr_nnps - std_dist_tfr_nnps, xmax=mean_dist_tfr_nnps + std_dist_tfr_nnps, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_tfr_nnps:.2f}')
    axes[2].annotate(f'skewness: {skew_dist_tfr_nnps:.2f}', xy=(-13000, 50),  bbox=dict(boxstyle="round", color='#f39c12' ,alpha=0.1), size = 10)
    axes[2].legend(loc='upper left')
    axes[2].set_ylabel('Nanopolish vs. Tailfindr', fontsize=12, fontweight='bold')
    axes[2].set_title('')  
    axes[2].set_yticklabels([])  

    
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.supxlabel('Distances', fontsize=12, fontweight='bold')
    plt.show()


def plot_end_ditances(df:pd.DataFrame): 

    dist_ann_tfr = df['tail_end_tailfindr'] - df['tail_end_annote']
    mean_dist_ann_tfr = dist_ann_tfr.mean()
    median_dist_ann_tfr = dist_ann_tfr.median() 
    std_dist_ann_tfr = dist_ann_tfr.std()
    var_dist_ann_tfr = dist_ann_tfr.var()
    skew_dist_ann_tfr = dist_ann_tfr.skew()


    dist_ann_nnps = df['tail_end_nanopolish'] - df['tail_end_annote']
    mean_dist_ann_nnps = dist_ann_nnps.mean()
    median_dist_ann_nnps = dist_ann_nnps.median()
    std_dist_ann_nnps = dist_ann_nnps.std()
    var_dist_ann_nnps = dist_ann_nnps.var()
    skew_dist_ann_nnps = dist_ann_nnps.skew()


    dist_tfr_nnps = df['tail_end_nanopolish'] - df['tail_end_tailfindr']
    mean_dist_tfr_nnps = dist_tfr_nnps.mean()
    median_dist_tfr_nnps = dist_tfr_nnps.median()
    std_dist_tfr_nnps = dist_tfr_nnps.std()
    var_dist_ann_nnps = dist_tfr_nnps.var()
    skew_dist_tfr_nnps = dist_tfr_nnps.skew()

    
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Distribution of Distances in Predicted and Annotated End Coordinates', fontsize=16, fontweight='bold')

    sns.histplot(dist_ann_tfr, kde=True, bins=30, color='#3498db', alpha=0.4, ax=axes[0])
    axes[0].axvline(mean_dist_ann_tfr, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_ann_tfr:.2f}') 
    axes[0].axvline(median_dist_ann_tfr, color='green', linestyle='--', label=f'median: {median_dist_ann_tfr:.2f}')
    axes[0].axvspan(xmin=mean_dist_ann_tfr - std_dist_ann_tfr, xmax=mean_dist_ann_tfr + std_dist_ann_tfr, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_ann_tfr:.2f}')
    axes[0].annotate(f'skewness: {skew_dist_ann_tfr:.2f}', xy=(-10000, 50),  bbox=dict(boxstyle="round", color='#3498db' ,alpha=0.1), size = 10)
    axes[0].legend(loc='upper left')
    axes[0].set_ylabel('Tailfindr vs. Annotation', fontsize=12, fontweight='bold')
    axes[0].set_title('')  
    axes[0].set_yticklabels([])  
    axes[0].set_xlabel('') 

    
    sns.histplot(dist_ann_nnps, kde=True, bins=30, color='#9b59b6', alpha=0.5, ax=axes[1])
    axes[1].axvline(mean_dist_ann_nnps, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_ann_nnps:.2f}') 
    axes[1].axvline(median_dist_ann_nnps, color='green', linestyle='--', label=f'median: {median_dist_ann_nnps:.2f}')
    axes[1].axvspan(xmin=mean_dist_ann_nnps - std_dist_ann_nnps, xmax=mean_dist_ann_nnps + std_dist_ann_nnps, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_ann_nnps:.2f}')
    axes[1].annotate(f'skewness: {skew_dist_ann_nnps:.2f}', xy=(-18000, 50),  bbox=dict(boxstyle="round", color='#9b59b6', alpha=0.1), size = 10)
    axes[1].legend(loc='upper left')
    axes[1].set_ylabel('Nanopolish vs. Annotation', fontsize=12, fontweight='bold')
    axes[1].set_title('') 
    axes[1].set_yticklabels([])  
    axes[1].set_xlabel('') 

    sns.histplot(dist_tfr_nnps, kde=True, bins=30, color='#f39c12', alpha=0.4, ax=axes[2])
    axes[2].axvline(mean_dist_tfr_nnps, color='#e74c3c', linestyle='--', label=f'mean: {mean_dist_tfr_nnps:.2f}') 
    axes[2].axvline(median_dist_tfr_nnps, color='green', linestyle='--', label=f'median: {median_dist_tfr_nnps:.2f}')
    axes[2].axvspan(xmin=mean_dist_tfr_nnps - std_dist_tfr_nnps, xmax=mean_dist_tfr_nnps + std_dist_tfr_nnps, color='#BCC6CC', alpha=0.2, label=f'mean ± std: {std_dist_tfr_nnps:.2f}')
    axes[2].annotate(f'skewness: {skew_dist_tfr_nnps:.2f}', xy=(-15000, 50),  bbox=dict(boxstyle="round", color='#f39c12' ,alpha=0.1), size = 10)
    axes[2].legend(loc='upper left')
    axes[2].set_ylabel('Nanopolish vs. Tailfindr', fontsize=12, fontweight='bold')
    axes[2].set_title('')  
    axes[2].set_yticklabels([]) 

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.supxlabel('Distances', fontsize=12, fontweight='bold')
    plt.show()


def plot_length_ditances(df:pd.DataFrame): 

    dist_ann_tfr = df['tail_length_tailfindr'] - df['tail_length_annote']
    mean_dist_ann_tfr = dist_ann_tfr.mean()
    median_dist_ann_tfr = dist_ann_tfr.median() 

    dist_ann_nnps = df['tail_length_nanopolish'] - df['tail_length_annote']
    mean_dist_ann_nnps = dist_ann_nnps.mean()
    median_dist_ann_nnps = dist_ann_nnps.median()

    dist_tfr_nnps = df['tail_length_nanopolish'] - df['tail_length_tailfindr']
    mean_dist_tfr_nnps = dist_tfr_nnps.mean()
    median_dist_tfr_nnps = dist_tfr_nnps.median()

    
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    fig.suptitle('Distribution of Distances in Predicted and Annotated Length of Polya Tail', fontsize=16, fontweight='bold')

    sns.histplot(dist_ann_tfr, kde=True, bins=30, color='#3498db', alpha=0.2, ax=axes[0])
    axes[0].axvline(mean_dist_ann_tfr, color='#e74c3c', linestyle='--', label=f'Mean: {mean_dist_ann_tfr:.2f}') 
    axes[0].axvline(median_dist_ann_tfr, color='green', linestyle='--', label=f'Median: {median_dist_ann_tfr:.2f}') #2ecc71
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Tailfindr vs. Annotation', fontsize=12, fontweight='bold')
    axes[0].set_title('')  
    axes[0].set_yticklabels([])  
    axes[0].set_xlabel('') 

    
    sns.histplot(dist_ann_nnps, kde=True, bins=30, color='#9b59b6', alpha=0.2, ax=axes[1])
    axes[1].axvline(mean_dist_ann_nnps, color='#e74c3c', linestyle='--', label=f'Mean: {mean_dist_ann_nnps:.2f}') 
    axes[1].axvline(median_dist_ann_nnps, color='green', linestyle='--', label=f'Median: {median_dist_ann_nnps:.2f}')
    axes[1].legend(loc='upper right')
    axes[1].set_ylabel('Nanopolish vs. Annotation', fontsize=12)
    axes[1].set_title('') 
    axes[1].set_yticklabels([])  
    axes[1].set_xlabel('') 

    sns.histplot(dist_tfr_nnps, kde=True, bins=30, color='#f39c12', alpha=0.2, ax=axes[2])
    axes[2].axvline(mean_dist_tfr_nnps, color='#e74c3c', linestyle='--', label=f'Mean: {mean_dist_tfr_nnps:.2f}') 
    axes[2].axvline(median_dist_tfr_nnps, color='green', linestyle='--', label=f'Median: {median_dist_tfr_nnps:.2f}')
    axes[2].legend(loc='upper right')
    axes[2].set_ylabel('Nanopolish vs. Tailfindr', fontsize=12, fontweight='bold')
    axes[2].set_title('')  
    axes[2].set_yticklabels([]) 

    plt.subplots_adjust(hspace=0.5)
    fig.supxlabel('X Values', fontsize=12, fontweight='bold')
    plt.show()


def multiple_boxplot(df: pd.DataFrame):

    df['distance_start_tfr_ann'] = df['tail_start_tailfindr'] - df['tail_start_annote'] 
    df['distance_start_nnps_ann'] = df['tail_start_nanopolish'] - df['tail_start_annote'] 
    
    df['distance_end_tfr_ann'] = df['tail_end_tailfindr'] - df['tail_end_annote'] 
    df['distance_end_nnps_ann'] = df['tail_end_nanopolish'] - df['tail_end_annote'] 
    
    df['distance_length_tfr_ann'] = df['tail_length_tailfindr'] - df['tail_length_annote'] 
    df['distance_length_nnps_ann'] = df['tail_length_nanopolish'] - df['tail_length_annote'] 
    
    melted_df = pd.melt(df, id_vars=['tail_start_annote', 'tail_end_annote', 'tail_length_annote'], 
                            value_vars=['distance_start_tfr_ann', 'distance_start_nnps_ann',
                                    'distance_end_tfr_ann', 'distance_end_nnps_ann',
                                    'distance_length_tfr_ann', 'distance_length_nnps_ann'],
                            var_name='Tool_Metric', 
                            value_name='Distance')
    
    melted_df['Metric'] = melted_df['Tool_Metric'].apply(lambda x: 'Start' if 'start' in x 
                                                         else 'End' if 'end' in x 
                                                         else 'Length')
    
    melted_df['Tool'] = melted_df['Tool_Metric'].apply(lambda x: 'Tailfindr' if 'tfr' in x 
                                                       else 'Nanopolish')
    
    plt.figure(figsize=(20, 15))
    sns.boxplot(x='Metric', y='Distance', hue='Tool', data=melted_df, palette={'Tailfindr': '#3498db', 'Nanopolish': '#ffa500'}, fill=False, gap=0.1, saturation=0.75, width=0.5)
    plt.title('Distribution of Distances for Start, End, and Length by Tool', fontsize=16, fontweight='bold')
    plt.xlabel('Metric (Start, End, Length)', fontsize=12, fontweight='bold')
    plt.ylabel('Distance = Tool - Annotation', fontsize=12, fontweight='bold')
    plt.legend(title='Tool', loc='upper right',  bbox_to_anchor=(1, 1))
    plt.show()

def multiple_violinplot(df:pd.DataFrame) : 

    df['distance_start_tfr_ann'] = df['tail_start_tailfindr'] - df['tail_start_annote'] 
    df['distance_start_nnps_ann'] = df['tail_start_nanopolish'] - df['tail_start_annote'] 
    
    df['distance_end_tfr_ann'] = df['tail_end_tailfindr'] - df['tail_end_annote'] 
    df['distance_end_nnps_ann'] = df['tail_end_nanopolish'] - df['tail_end_annote'] 
    
    df['distance_length_tfr_ann'] = df['tail_length_tailfindr'] - df['tail_length_annote'] 
    df['distance_length_nnps_ann'] = df['tail_length_nanopolish'] - df['tail_length_annote'] 
    
    melted_df = pd.melt(df, id_vars=['tail_start_annote', 'tail_end_annote', 'tail_length_annote'], 
                            value_vars=['distance_start_tfr_ann', 'distance_start_nnps_ann',
                                    'distance_end_tfr_ann', 'distance_end_nnps_ann',
                                    'distance_length_tfr_ann', 'distance_length_nnps_ann'],
                            var_name='Tool_Metric', 
                            value_name='Distance')
    
    melted_df['Metric'] = melted_df['Tool_Metric'].apply(lambda x: 'Start' if 'start' in x 
                                                         else 'End' if 'end' in x 
                                                         else 'Length')
    
    melted_df['Tool'] = melted_df['Tool_Metric'].apply(lambda x: 'Tailfindr' if 'tfr' in x 
                                                       else 'Nanopolish')
    
    plt.figure(figsize=(20, 15))
    sns.violinplot(x='Metric', y='Distance', hue='Tool', data=melted_df, 
               palette={'Tailfindr': '#3498db', 'Nanopolish': '#f39c12'}, 
               split=True, inner="box", scale="width", fill=False, )
    
    plt.title('Distribution of Distances for Start, End, and Length by Tool', fontsize=16, fontweight='bold')
    plt.xlabel('Metric (Start, End, Length)', fontsize=12, fontweight='bold')
    plt.ylabel('Distance = Tool - Annotation', fontsize=12, fontweight='bold')
    plt.legend(title='Tool', loc='upper right',  bbox_to_anchor=(1, 1))
    plt.show()

def multiple_scatter(df:pd.DataFrame) : 
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    #fig.suptitle('Distances of Start and End in Tools vs. Values of Annotation', fontsize=20, fontweight='bold')
    
    df['distance_start_tfr_ann'] = df['tail_start_tailfindr'] - df['tail_start_annote'] 
    df['distance_start_nnps_ann'] = df['tail_start_nanopolish'] - df['tail_start_annote'] 
    
    mean_dist_start_tfr_ann = df['distance_start_tfr_ann'].mean()
    mean_dist_start_nnps_ann = df['distance_start_nnps_ann'].mean()

    melted_df_start = df.melt(id_vars='tail_start_annote', 
                    value_vars=['distance_start_tfr_ann', 'distance_start_nnps_ann'], 
                    var_name='Tool_Start', value_name='Distance_Start')

    palette = {'distance_start_tfr_ann': '#3498db', 'distance_start_nnps_ann': '#f39c12'} #f39c12
    
    sns.scatterplot(data=melted_df_start, x='tail_start_annote', y='Distance_Start', hue='Tool_Start', palette=palette, ax=axes[0], 
                   legend=False, markers={'distance_start_tfr_ann' : 'X', 'distance_start_nnps_ann' : 'o'}, s=95,  alpha=0.5)
    axes[0].axhline(mean_dist_start_tfr_ann, color='#ff2c2c', linestyle='--', label=f'Tailfindr: {mean_dist_start_tfr_ann:.2f}')
    axes[0].axhline(mean_dist_start_nnps_ann, color='#4cbb17', linestyle='--', label=f'Nanopolish: {mean_dist_start_nnps_ann:.2f}')
    axes[0].set_ylim(-100, 100)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    #axes[0].set_xlabel('Start Values of Annotation', fontsize = 16, fontweight='bold', labelpad=10.0)
    #axes[0].legend(title='Mean', loc='upper right', fontsize=14, title_fontproperties={'weight': 'bold'})
    
    # -----------------------

    df['distance_end_tfr_ann'] = df['tail_end_tailfindr'] - df['tail_end_annote'] 
    df['distance_end_nnps_ann'] = df['tail_end_nanopolish'] - df['tail_end_annote'] 
    
    mean_dist_end_tfr_ann = df['distance_end_tfr_ann'].mean()
    mean_dist_end_nnps_ann = df['distance_end_nnps_ann'].mean()

    melted_df_end = df.melt(id_vars='tail_end_annote', 
                    value_vars=['distance_end_tfr_ann', 'distance_end_nnps_ann'], 
                    var_name='Tool_End', value_name='Distance_End')

    palette = {'distance_end_tfr_ann': '#3498db', 'distance_end_nnps_ann': '#f39c12'} 
  
    sns.scatterplot(data=melted_df_end, x='tail_end_annote', y='Distance_End', hue='Tool_End', palette=palette, ax=axes[1], 
                   legend=False, markers={'distance_end_tfr_ann' : 'X', 'distance_end_nnps_ann' : 'o'}, s=95,  alpha=0.5)
    axes[1].axhline(mean_dist_end_tfr_ann, color='#ff2c2c', linestyle='--', label=f'Tailfindr: {mean_dist_end_tfr_ann:.2f}')
    axes[1].set_ylim(-100, 100)
    axes[1].axhline(mean_dist_end_nnps_ann, color='#4cbb17', linestyle='--', label=f'Nanopolish: {mean_dist_end_nnps_ann:.2f}')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    #axes[1].set_xlabel('End Values of Annotation', fontsize = 16, fontweight='bold', labelpad=10.0)
    #axes[1].legend(title='Mean', loc='upper right', fontsize=14, title_fontproperties={'weight': 'bold'})
    
    # ------------------------

    """
    df['distance_length_tfr_ann'] = df['tail_length_tailfindr'] - df['tail_length_annote'] 
    df['distance_length_nnps_ann'] = df['tail_length_nanopolish'] - df['tail_length_annote'] 
    
    mean_dist_length_tfr_ann = df['distance_length_tfr_ann'].mean()
    mean_dist_length_nnps_ann = df['distance_length_nnps_ann'].mean()

    melted_df_length = df.melt(id_vars='tail_length_annote', 
                    value_vars=['distance_length_tfr_ann', 'distance_length_nnps_ann'], 
                    var_name='Tool_Length', value_name='Distance_Length')

    palette = {'distance_length_tfr_ann': '#3498db', 'distance_length_nnps_ann': '#f39c12'} #f39c12
  
    sns.scatterplot(data=melted_df_length, x='tail_length_annote', y='Distance_Length', hue='Tool_Length', palette=palette, ax=axes[2], 
                   legend=False, markers={'distance_length_tfr_ann' : 'X', 'distance_length_nnps_ann' : 'o'}, s=95,  alpha=0.5)
    axes[2].axhline(mean_dist_length_tfr_ann, color='red', linestyle='--', label=f'Tailfindr: {mean_dist_length_tfr_ann:.2f}')
    axes[2].axhline(mean_dist_length_nnps_ann, color='green', linestyle='--', label=f'Nanopolish: {mean_dist_length_nnps_ann:.2f}')
    axes[2].set_ylabel('')
    axes[2].set_xlabel('Length Values of Annotation', fontsize = 12, fontweight='bold')
    axes[2].legend(title='Mean') 
    """
    
    #plt.subplots_adjust(hspace=0.1, wspace=0.2, left=0.07)

    #fig.supylabel('Distance = Tool - Annotation', fontsize=16, fontweight='bold')
    #handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Tailfindr', alpha=0.5),
    #           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=10, label='Nanopolish', alpha=0.5)]
    
    #fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=14, title_fontproperties={'weight': 'bold'})
    
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    
    #plt.savefig('./comparisons_plots/main_scatter.png')
    
    plt.show()





def check_length_correlation(df:pd.DataFrame, column1:str, column2:str, length_column:str):
        
    distance = df[column1] - df[column2]
    
    correlation = distance.corr(df[length_column])
    
    plt.figure(figsize=(14, 7))
    sns.regplot(y=distance, x=df[length_column], scatter_kws={'s':50}, line_kws={"color":"orange"})
    plt.title(f'Scatter plot of Differences in {column1} and {column2} vs. Tail Length predicted by tool \nCorrelation: {correlation:.2f}')
    plt.ylabel(f'Difference between {column1} and {column2}')
    plt.xlabel('Tail Length')    
 
def save_dataframe(df:pd.DataFrame): 
    
    path = set_working_directory()
    file_path = path + '/' + 'main_comparison_table.tsv'
    df.to_csv(file_path, sep='\t', encoding='utf-8', index=False, header=True)
    print('successfully saved the dataframe.')


def main():
    
    # setup

    set_working_directory()
    
    # load

    ann_coord = load_sort_csv('annotations/all_annotated_coordinates.csv')
    ann_coord.columns = ['read_id' ,'tail_start_annote', 'tail_end_annote', 'alt_tail_start', 'alt_tail_end'] # reset the col names 
    ann_coord['tail_length_annote'] = ann_coord['tail_end_annote'] - ann_coord['tail_start_annote'] # add tail_column based on tail_start and tail_end
    ann_coord.drop(['alt_tail_start', 'alt_tail_end'], axis=1, inplace=True)   

    tailfindr_coord = load_sort_csv('tailfindr_polya/tailfindr_after_basecalling/new_tailfindr/ecoli_164_subset_tails.csv')
    tailfindr_coord.columns = ['read_id', 'tail_start_tailfindr', 'tail_end_tailfindr', 'samples_per_nt', 'tail_length_tailfindr','file_path']
    tailfindr_coord.drop(['samples_per_nt', 'file_path'], axis=1, inplace=True)
    
    nnps_coord = load_sort_csv('nanopolish_polya/ecoli_polya_all_reads.tsv')
    nnps_coord.columns = ['read_id', 'contig', 'position', 'leader_start', 'adapter_start',
                           'tail_start_nanopolish', 'tail_end_nanopolish', 'read_rate', 'tail_length_nanopolish', 'qc_tag_nanopolish']
    nnps_coord.drop(['read_rate', 'adapter_start', 'leader_start', 'position', 'contig'], axis=1, inplace=True)

    #dorado_lengths = load_sort_csv('dorado_output/pt_values.csv')

    # merge 
     
    merged_ann_tailfindr = pd.merge(ann_coord, tailfindr_coord, on='read_id') # 163 rows 
    merged_ann_nnps = pd.merge(ann_coord, nnps_coord, on='read_id') # 100 rows 
    merged_tfr_nnps = pd.merge(tailfindr_coord, nnps_coord, on='read_id') # 95 rows
    
    # merged_ann_dorado = pd.merge(ann_coord, dorado_lengths, on='read_id', suffixes=('_annote', '_dorado'))
    
    # final dataframe 95 rows
    merged_tfr_nnps_ann = pd.merge(merged_ann_tailfindr, nnps_coord, on='read_id')  
    merged_tfr_nnps_ann = merged_tfr_nnps_ann.drop_duplicates(subset=['read_id'], keep='first')

    

    #plot_start_ditances(merged_tfr_nnps_ann)
    
    #plot_end_ditances(merged_tfr_nnps_ann)

    #plot_length_ditances(merged_tfr_nnps_ann)

    #
    multiple_scatter(merged_tfr_nnps_ann)

    ##save_dataframe(merged_tfr_nnps_ann)
    
    #multiple_boxplot(merged_tfr_nnps_ann)
    
    #multiple_violinplot(merged_tfr_nnps_ann)
    
    
    # main dataset for model comparison
    # merged_tfr_nnps_ann.to_csv('annotations/merged_tfr_nnps_ann_readids.csv', mode='w', header=True, index=False)

    

if __name__ == "__main__":
    main()


