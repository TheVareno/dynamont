
import pysam #type: ignore 

samfile = pysam.AlignmentFile('/data/fass5/projects/hv_rna_mod/data/basecalled/psU/rna_hac_70bps/psU-RNA_20201103_FAO12159.bam', mode='rb', check_sq=False)

for read in samfile.fetch(until_eof=True): 
        
        read_name = read.query_name 
        tags = dict(read.tags)  
        print(f'read name ---- {read.query_name}') 
        print(f'read length ---- {read.query_length}') 
        
        for pair in tags:
                print(f'{pair} ---- {tags[pair]}')

        break