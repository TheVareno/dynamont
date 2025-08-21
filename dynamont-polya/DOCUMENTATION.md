
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

Manual annotation was guided by the study [Nanopolish polyA](https://www.nature.com/articles/s41592-019-0617-2#Sec1), specifically the plots provided in the [Supplemental Materials](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-019-0617-2/MediaObjects/41592_2019_617_MOESM1_ESM.pdf), "Model Description" section, Figure 1, which enabled approximate annotation of region boundaries by visual inspection.  

After annotation, the signal values of each region were extracted, filtered, normalized, and concatenated across more than 30 reads. These aggregated signals were then fitted to various probability density functions (PDFs), and the PDF with the highest likelihood was chosen to represent each region.  

Using these PDFs, the Baum–Welch algorithm was applied to *E. coli* read data with a synthetic poly(A) tail in order to estimate the transition probabilities between regions.  

The defined regions were modeled as the states of a one-dimensional Hidden Markov Model (HMM). The fitted PDFs served as emission probabilities, while the learned transition probabilities defined the likelihood of switching between states. This HMM framework enabled segmentation of raw RNA signals into their respective key regions.  

With the start and end of the poly(A) region identified, the next step was to develop a method for estimating poly(A) tail length.  



