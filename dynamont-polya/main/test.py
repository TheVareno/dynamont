
import pysam #type: ignore 

samfile = pysam.AlignmentFile('/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data/FAX28269.bam', mode='rb', check_sq=False)

read_count = 0 
for read in samfile.fetch(until_eof=True): 

        print(read.query_name) 
        read_count = read_count + 1 
        
        tags = dict(read.tags)  
        
        #print(f'read name ---- {read.query_name}') 
        #print(f'read length ---- {read.query_length}') 
        
        #for pair in tags:
        #        print(f'{pair} ---- {tags[pair]}')
        
        if read_count > 100:
                break 






