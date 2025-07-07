
"""
Author : Hadi Vareno 
Email : mohammad.noori.vareno@uni-jena.de / hadivareno@gmail.com
Github:  https://github.com/TheVareno
"""

"""
This scripts: 
- 

"""

import os 
import pysam # type: ignore

def set_current_dir(path: str)-> None:
    if os.getcwd != path : 
        os.chdir(path)


def main(): 
    set_current_dir('/home/hi68ren/Dokumente/ProjektModul/Implementation/main_data/ecoli/annotation_comparison')
    
    sam_data = pysam.AlignmentFile("dorado_output/ecoli_dorado_calls_polya.sam", "r")
    print(sam_data) 


if __name__ == '__main__':
    main() 

