

INPUT_FILE="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data/FAX28269_36c48ee6_b042d0cd_0.fast5"
SEQUINS_INPUT="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/Sequins"
ALT_INPUT_DIR="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/alt_input"
INPUT_DIR="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data"
INPUT_TEST="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/test_input"
OUTPUT_DIR="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/output_data"
ALT_OUTPUT_DIR="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/alt_output"
SEQUINS_OUTPUT="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/Sequins_output"
SUMMARY_FILE="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/alt_input/sequencing_summary_0.txt"
BAM_FILE="/home/hi68ren/Dokumente/dynamont-fork/dynamont/dynamont-polya/input_data/FAX28269.bam"


time python polyA.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --bam_file "$BAM_FILE"


# last errorless try - run time: 
# TODO: monitor memory usage ! 
# real	29m35.848s
# user	210m29.969s
# sys	1m12.489s



"""
Runtime report ! 

Am 05.08: 
real	33m37.327s
user	129m24.504s
sys	0m32.869s

"""


# time python polyA.py --input_dir ../input_data/FAX28269_36c48ee6_b042d0cd_0.fast5 --output_dir ../output_data --bam_file ../input_data/FAX28269.bam