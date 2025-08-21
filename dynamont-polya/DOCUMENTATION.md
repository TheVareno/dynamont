
# Poly(A) Tail Segmentation & Length Estimation (HMM) 

- Branch: dynamont/polyA
- Constributors:
  - Hadi Vareno (hadivareno@gmail.com)  
  - Jannes Spangenberg (jannes.spangenberg@uni-jena.de)  
- Status:
  - MVP complete
  - Training & inference run end-to-end on test data
  - optimization in progress.

---

## Introduction  

This project focuses on identifying the poly(A) region within raw signal data from Oxford Nanopore direct RNA sequencing and estimating the length of the poly(A) tail.  

As an initial step, 100 random reads were selected for manual annotation of the start and end of key regions in the raw signal data. The annotated regions include:  

- **start**: opening pore signals  
- **leader**: signals corresponding to the splint adapter  
- **adapter**: signals corresponding to the sequencing adapter  
- **polyA**: signals corresponding to the poly(A) tail  
- **transcript**: signals corresponding to the transcript body  

Manual annotation was guided by the study [Nanopolish polyA](https://www.nature.com/articles/s41592-019-0617-2#Sec1), specifically the plots provided in the [Supplemental Materials](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-019-0617-2/MediaObjects/41592_2019_617_MOESM1_ESM.pdf), "Model Description" section, Figure 1, which enabled approximate annotation of region boundaries by visual inspection.  

After annotation, the signal values of each region were extracted, filtered, normalized, and concatenated across more than 30 reads. These aggregated signals were then fitted to various probability density functions (PDFs), and the PDF with the highest likelihood was chosen to represent each region.  

Using these PDFs, the Baum–Welch algorithm was applied to *E. coli* read data with a synthetic poly(A) tail in order to estimate the transition probabilities between regions.  

The defined regions were modeled as the states of a one-dimensional Hidden Markov Model (HMM). The fitted PDFs served as emission probabilities, while the learned transition probabilities defined the likelihood of switching between states. This HMM framework enabled segmentation of raw RNA signals into their respective key regions.  

With the start and end of the poly(A) region identified, the next step was to develop a method for estimating poly(A) tail length.   

--- 

## Repository Structure (Directory and File Overview)  

 in Branch: dynamont/polyA -> under the dirrectiory of **/dynamont-polya**:    


- **build_model_pipeline/**  
  - **model_building_files/** → log probabilities/ trained parameters saved during model building   
  - `annotate_signal.py` → scirpts for manuall annotation of raw signal data     
  - `coordinations_analysis.py` →  visaulization of Nanopolish and TailfindR performance  
  - `model_performance_comparison.py` → comparison of HMM model performance  
  - `pdf_estimation.py` → fitt and estimate PDF for each region  
  - `pysam_utils.py` → using pysam to extract data from Basecaller (Dorado) output
  - `run_commands.sh` → run the script with given parameter
  - `tail_findr.R` → rund tailfindr package in R script


- **input_data/**  
  - `reads.fast5` → example input file  
  - `sequencing_summary_0.txt` → summary of experiment (potential usage for length estimation)   

- **output_data/**  
  - `output_test.csv` → output of read id, poly(A) borders with estimated length for each given read id in CSV format    

- **main/**  
  - `argparse.hpp` → containing script to parse aguments  
  - `bam_analysis.py` → analysis of basecaller output in bam format for poly(A) length estimation  
  - `polyA.cpp` → compute the forward and backward alogortihm, also trains Hidden Markov Model using Baum–Welch algorithm  
  - `polyA.hpp` → the header file main computation, containing all the functions needed for segmentation and traning the model    
  - `polyA.py` → reads the given fast5 / pod5 / slow5 input read, send signal data to polyA.cpp using stdout    
  - `polyA.sh` → runs the polyA.py with all given arguments 
  - `utils.cpp` → contains function helping forward backward calculations  
  - `utils.hpp` → the header file   
  - `polyA` → the binary file of compiled version of polyA.cpp   

- `requirements.txt` → required packages to install   
- `DOCUMENTATION.md` → containing all the structures    
 
---

## Usage  

The algorithm can be executed via the main script **`polyA.py`**.  
The following parameters must be provided:  

```bash
python polyA.py \
  --input_dir <PATH_TO_INPUT_DIRECTORY> \
  --output_dir <PATH_TO_OUTPUT_DIRECTORY> \
  --bam_file <PATH_TO_BAM_FILE>
```
---

## Poly(A) Tail Length Estimation

The estimation of poly(A) tail length is based on the predicted start and end coordinates of the poly(A) region in the raw signal.

The key intuition is that the poly(A) tail corresponds to a continuous stretch of signal values, and its nucleotide length can be inferred by normalizing the signal length with the average sampling rate per nucleotide.

---

### Step 1: Poly(A) signal span

Let
- $s_{\text{start}}$ = predicted start coordinate of the poly(A) region in the raw signal
- $s_{\text{end}}$ = predicted end coordinate of the poly(A) region in the raw signal

The number of raw signal samples corresponding to the poly(A) region is:

$$
L_{\text{signal}}^{\text{poly(A)}} = s_{\text{end}} - s_{\text{start}}
$$

---

### Step 2: Average samples per nucleotide

For each transcript, we calculate the average number of signal samples per nucleotide.

Let
- $L_{\text{signal}}^{\text{transcript}}$ = total number of signal samples aligned to the transcript
- $L_{\text{nt}}^{\text{transcript}}$ = transcript length in nucleotides (obtained from the BAM file output by Dorado)

Then:

$$
\text{samples\_per\_nt} = \frac{L_{\text{signal}}^{\text{transcript}}}{L_{\text{nt}}^{\text{transcript}}}
$$

---

### Step 3: Estimate poly(A) tail length in nucleotides

Finally, the estimated poly(A) tail length is given by:

$$
L_{\text{nt}}^{\text{poly(A)}} = \frac{L_{\text{signal}}^{\text{poly(A)}}}{\text{samples\_per\_nt}}
$$

This formula ensures that the length of the poly(A) tail is corrected for differences in sampling rate across different reads.

## Benchmark Dastaset 


--- 

## Output

The output is a tabular file where each row corresponds to a single read from the input dataset.  
It contains the poly(A) start and end positions within the raw signal, along with the estimated poly(A) tail length.  

| Read ID        | Poly(A) Start | Poly(A) End | Estimated Poly(A) Length |
|----------------|---------------|-------------|--------------------------|
| read_00123abc  | 1050          | 1450        | 400                      |

*Example output row. In practice, the output table will typically contain thousands of rows (e.g., ~4000 rows for a dataset of 4000 FAST5 reads).*  


--- 

## Other References  






