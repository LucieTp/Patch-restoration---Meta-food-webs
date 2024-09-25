# Script to run the restoration experiment on meta food webs

To replicate the analysis, you will first need to clone the conda environment or make sure to have the right version of python and all modules as described in the main text and conda_environment.txt file.

To run the first .py script 1.0.RestorationSimulations.py you will likely need to 
(1) update directory names to those one your personal computer in the .py file
(2) create the right folders including 
>> 15Patches/Heterogeneous/Sim0; 15Patches/Heterogeneous/Sim1; etc for simulations [0,1,2,6,13]
>> 15Patches/Homogeneous/Sim0; 15Patches/Heterogeneous/Sim1; etc for simulations [0,1,2,6,13]

These simulations take a long time (weeks) to run, so will likely be done in batches. 
This is why I created the 1.1.WritingFileNames.py script, which searches the directory where population dynamics are stored 
and lists all file names that have already run. This feedbacks to 1.0.RestorationSimulations.py so that it knows it doesn't need to run again these simulations again. 

note: I did this in a text file because I was running simulations from the SLURM cluster and didn't have space to store all simulations there simultaneously. 
/!\ You will however, always need to keep the dynamics from the initial simulations as these are the baseline files to run all subsequent experiments.

Once all simulations have run 
(3) create summary statistics files (.csv) for each stage of restoration using 2.CreateSummaryFiles.py
(4) 3.Analysis.py has the code to rerun all analysis from the manuscript and recreate graphs