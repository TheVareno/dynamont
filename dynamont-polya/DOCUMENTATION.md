
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
  - `argparse.hpp` → header file for parsing command-line arguments, providing a clean interface to handle user inputs and options
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

# Poly(A) Tail Length Estimation

The estimation of poly(A) tail length is based on the predicted start and end coordinates of the poly(A) region in the raw signal. The key intuition is that the poly(A) tail corresponds to a continuous stretch of signal values, and its nucleotide length can be inferred by normalizing the signal length with the average sampling rate per nucleotide.

## Step 1: Poly(A) Signal Span

Let:
- $s_{\text{start}}$ = predicted start coordinate of the poly(A) region in the raw signal
- $s_{\text{end}}$ = predicted end coordinate of the poly(A) region in the raw signal

The number of raw signal samples corresponding to the poly(A) region is:

$$L_{\text{signal}}^{\text{poly(A)}} = s_{\text{end}} - s_{\text{start}}$$

## Step 2: Average Samples per Nucleotide

For each transcript, we calculate the average number of signal samples per nucleotide.

Let:
- $L_{\text{signal}}^{\text{transcript}}$ = total number of signal samples aligned to the transcript
- $L_{\text{nt}}^{\text{transcript}}$ = transcript length in nucleotides (obtained from the BAM file output by Dorado)

Then:

$$\text{samples per nt} = \frac{L_{\text{signal}}^{\text{transcript}}}{L_{\text{nt}}^{\text{transcript}}}$$

## Step 3: Estimate Poly(A) Tail Length in Nucleotides

Finally, the estimated poly(A) tail length is given by:

$$L_{\text{nt}}^{\text{poly(A)}} = \frac{L_{\text{signal}}^{\text{poly(A)}}}{\text{samples per nt}}$$

### Summary

1. Identifying the signal span of the poly(A) region
2. Calculating the average signal sampling rate per nucleotide
3. Normalizing the poly(A) signal length by the sampling rate

The resulting estimate $L_{\text{nt}}^{\text{poly(A)}}$ represents the poly(A) tail length in nucleotides.

--- 
 
## Benchmark Dataset  

For evaluation of the poly(A) length estimation, the benchmark dataset can be used which is described in the preprint study:  
- **Preprint link**: [https://www.biorxiv.org/content/10.1101/2024.10.25.620206v1.full.pdf]  
- **Dataset link**: [https://doi.org/10.5524/102736]
- **Reference of Sequins** [https://github.com/abcdtree/Sequins_QC/tree/main/ref]

To obtain access to the raw data, the corresponding author of the study was contacted directly, and the dataset was kindly shared for benchmarking purposes.  

In this study, the poly(A) tail estimation was benchmarked among 4 exsiting tools: 
 - Tailfindr (mathematical approach to preprocess the signal data)
 - Nanopolish polya (HMM based approach)
 - Dorado polya-estimation (deep learning based approach)
 - BoostNano (deep learning based approach)
   
The link above contains all the FAST5 data of spiked-in RNA and CSV performances, outputed by each tool 4 tools.  

### Dataset Description  

The benchmark data consists of *in vitro* transcribed (IVT) RNA molecules with controlled poly(A) tails of known lengths. The poly(A) tails were designed in two groups:  

- **Short poly(A) group**: fixed tail length of ~30 nucleotides  
- **Long poly(A) group**: fixed tail length of ~60 nucleotides  

The dataset contains nanopore direct RNA sequencing reads from these spiked RNA molecules, providing a **gold standard** for evaluating poly(A) identification and length estimation.  

This design allows a direct comparison between predicted poly(A) lengths from the algorithm and the known ground truth lengths, making it an ideal dataset for benchmarking.

--- 

## Output

The output is a tabular file where each row corresponds to a single read from the input dataset.  
It contains the poly(A) start and end positions within the raw signal, along with the estimated poly(A) tail length.  

| Read ID        | Poly(A) Start | Poly(A) End | Estimated Poly(A) Length |
|----------------|---------------|-------------|--------------------------|
| read_00123abc  | 1050          | 1450        | 400                      |

*Example output row. In practice, the output table will typically contain thousands of rows (e.g., ~4000 rows for a dataset of 4000 FAST5 reads).*  


--- 

## Details on latest Performance (updated on 21.08.2025) 


--- 

## Other References  

- **BoostNano** repository [https://github.com/haotianteng/BoostNano]
- **Nanopolish** repository [https://github.com/jts/nanopolish]
- **Review** study[https://pubmed.ncbi.nlm.nih.gov/35769001] for measuring poly(A) tail length  
- **Nanopolish**[https://www.nature.com/articles/s41592-019-0617-2?fromPaywallRec=false#Sec12] orginal study
- **TailfindR**[https://pmc.ncbi.nlm.nih.gov/articles/PMC6800471/#s05] orginal study
- **Dorado**[https://github.com/nanoporetech/dorado/blob/release-v1.0/documentation/PolyTailConfig.md] bam data description










