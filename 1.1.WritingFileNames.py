# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:45:06 2024

@author: lucie.thompson
"""

# %% writing file names to csv file

import os
import csv

# Define the directory you want to list files from
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow')
init_15P_files = [i for i in os.listdir() if 'InitialPopDynamics' in i or 'NotStabilised_Init' in i]


# Define the CSV file name
csv_file_name = 'D:/TheseSwansea/Patch-Models/outputs/15Patches/init_files_15Patches_narrow.csv'

# Write the file names to a CSV file
with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['File Name'])
    
    # Write the file namesf
    for file_name in init_15P_files:
        csvwriter.writerow([file_name])

# Control
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/narrow')
control_15P_files = [i for i in os.listdir() if '.pkl' in i and 'CONTROL' in i]

csv_file_name = 'D:/TheseSwansea/Patch-Models/outputs/15Patches/control_files_15Patches_narrow.csv'

with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['File Name'])
    
    # Write the file namesf
    for file_name in control_15P_files:
        csvwriter.writerow([file_name])


## restored
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow')
het_15P_files_invasion = [i for i in os.listdir() if 'patchImproved' in i or 'NotStabilised_restoration' in i]

csv_file_name = 'D:/TheseSwansea/Patch-Models/outputs/15Patches/improved_files_15Patches_narrow.csv'

with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['File Name'])
    
    # Write the file namesf
    for file_name in het_15P_files_invasion:
        csvwriter.writerow([file_name])