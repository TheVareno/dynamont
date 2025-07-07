
"""
Author : Hadi Vareno 
Email : mohammad.noori.vareno@uni-jena.de
Github:  https://github.com/TheVareno
"""


Sys.setenv(HDF5_PLUGIN_PATH="/usr/local/hdf5/lib/plugin")

library(rhdf5)
library(tailfindr)
library(rbokeh)

fast5_dir <- "Documents/Projects/ProjektModul/Implementation/polyt_read/"

print(list.files(fast5_dir, pattern = "*.fast5"))

df <- find_tails(
  fast5_dir = fast5_dir,
  save_dir = fast5_dir,
  csv_filename = 'test_tails.csv',
  num_cores = 2,
  save_plots = TRUE,
  plotting_library = 'rbokeh'
)

output <- read.csv("Documents/Projects/ProjektModul/Implementation/polyt_read/test_tails.csv")


print(h5ls(file = "Documents/Projects/ProjektModul/Implementation/polya_reads/5.fast5", recursive = TRUE))

