# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:46:04 2024

@author: lucie.thompson
"""

#%% Create summary stats

import os
import pandas as pd


os.chdir('D:/TheseSwansea/SFT/Script')
import FunctionsAnalysisRestorationDonut as fn

P = 15
Stot = 100
C = 0.1

# %%%% initial dynamics

# 24/09 - updated mfcl

# intial population dynamics - normal
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/narrow/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]

# run summary statistics function (above)
res15_init, FW15_init, FW15_chara_init = fn.summarise_initial_pop_dynamics(list_files=init_15P_files, nb_patches=P)

FW15_init['simulation_length_years'] = FW15_init['simulation_length']/(60*60*24*365)

res15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-narrow_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
FW15_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-narrow-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
FW15_chara_init.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-narrow-RegionalFoodWebChara_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')


# %%%% control

# to get final biomass
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')
res15_init_normal = pd.read_csv(f'ResultsInitial-narrow_sim_{P}Patches_{Stot}sp_{C}C_02092024.csv')


# Control
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/narrow')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/narrow/'+i for i in os.listdir() if '.pkl' in i and 'invasion' in i]

res15, FW15, FW15_chara = fn.summarise_pop_dynamics(list_files=control_invasion_15P_files, nb_patches=P, 
                                        initial_summary=res15_init_normal)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-narrow-CornerPatch-handmade_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-narrow-CornerPatch-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
FW15_chara.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-narrow-invasion-CornerPatch-RegionalFoodWebChara_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')



# %%%% restoration

# to get final biomass
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')
res15_init_normal = pd.read_csv(f'ResultsInitial-narrow_sim_{P}Patches_{Stot}sp_{C}C_27042024.csv')


P = 15
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow')
het_15P_files_invasion = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow/'+i for i in os.listdir() if '.pkl' in i and 'patchImproved' in i]


# run summary statistics function (above)
res15, FW15, FW15_chara = fn.summarise_pop_dynamics(list_files=het_15P_files_invasion, nb_patches=P, 
                                        initial_summary=res15_init_normal)

FW15['simulation_length_years'] = FW15['simulation_length']/(60*60*24*365)

res15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
FW15.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
FW15_chara.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch-RegionalFoodWebChara_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv')
