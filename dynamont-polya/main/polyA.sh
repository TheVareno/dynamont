

INPUT_FILE="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data/FAX28269_36c48ee6_b042d0cd_0.fast5"
OUTPUT_DIR="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/output_data"
BAM_FILE="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data/FAX28269.bam"

echo "Starting polyA.py script..."

time python ./polyA.py \
    --input_dir "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --bam_file "$BAM_FILE"

echo "Script finished successfully."

# -s /home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data/sequencing_summary_0.txt
 
# last errorless try - run time: 

# TODO: monitor memory usage ! 
# real	29m35.848s
# user	210m29.969s
# sys	1m12.489s

# time python polyA.py --input_dir ../input_data/FAX28269_36c48ee6_b042d0cd_0.fast5 --output_dir ../output_data --bam_file ../input_data/FAX28269.bam