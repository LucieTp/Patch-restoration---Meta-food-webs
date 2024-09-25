# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:08:31 2023

Script for running analysis of restoration experiment on meta food webs

@author: lucie.thompson
"""



#%% Loading modules and global variables


import pandas as pd
## pandas needs to be version 1.5.1 to read the npy pickled files with np.load() 
## created under this version 
## this can be checked using pd.__version__ and version 1.5.1 can be installed using
    ## pip install pandas==1.5.1 (how I did it on SLURM - 26/03/2024)
## if version 2.0 or over yields a module error ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'
## I think to solve this, would need to run the code and save files under version > 2.0 of pandas
## or manually create this pandas.core.index.numerical function see issue in https://github.com/pandas-dev/pandas/issues/53300

import matplotlib.pyplot as plt 

import numpy as np # Numerical objects

import os

import seaborn as sb

import pickle

os.chdir('D:/TheseSwansea/SFT/Script')
import FunctionsAnalysisRestorationDonut as fn


f = "D:/TheseSwansea/Patch-Models/S100_C10/StableFoodWebs_55persist_Stot100_C10_t10000000000000.npy"
stableFW = np.load(f, allow_pickle=True).item()

palette_colors = sb.color_palette('coolwarm', 8)
sb.palplot(palette_colors)
plt.show()


Stot = 100 # initial pool of species
P = 15 # number of patches
C = 0.1 # connectance
tmax = 10**12 # number of time steps
d = 1e-8
# FW = fn.nicheNetwork(Stot, C)

# S = np.repeat(Stot,P) # local pool sizes (all patches are full for now)
S = np.repeat(round(Stot*1/3),P) # initiate with 50 patches


## create landscape 
extentx = [0,0.4]
radius_max = 0.1

## surface of the circle
((radius_max**2)*np.pi)/(0.5*0.5)


## make maps of each landscape and save:

# for seed_index in [0,1,3,2,4,5]:

#     seed = np.arange(P*seed_index, P*(seed_index + 1))
#     coords = fn.create_landscape(P = P, extent = extentx, radius_max = radius_max, nb_center = 5, seed = seed)
    
    
#     plt.figure(figsize=(5,5))
#     sb.scatterplot(data = coords, x = 'x',y = 'y', hue = 'Patch', style = 'position', palette = palette_colors)
#     plt.scatter(x = np.mean(extentx), y = np.mean(extentx), s = 100, marker = 'X', c = 'black') # center of the landscape
#     plt.legend(bbox_to_anchor = [1,1.1])
#     plt.ylim(0,0.4)
#     plt.xlim(0,0.4)
#     plt.savefig(f'D:/TheseSwansea/SFT/Figures/seed{seed_index}-Landscape.png', dpi = 400, 
#                 bbox_inches = 'tight')



## create 'landscape_characteristic' which has distance to invaded patch,
## distance to center, mean distance to other patches

## landscape 5 has 4 identical patches - removed from analysis
landscape_characteristics = pd.DataFrame()

for seed_index in [0,1,3,2,4]:

    seed = np.arange(P*seed_index, P*(seed_index + 1))
    coords = fn.create_landscape(P = P, extent = extentx, radius_max = radius_max, nb_center = 5, seed = seed)

    distance_left = np.array(((coords['x'] - extentx[0])**2 + (coords['y'] - extentx[1])**2)**(1/2))
    patch_to_invade = np.argmin(distance_left)
    coords_patch_to_invade = coords[coords['Patch'] == patch_to_invade]
    
    dist = fn.get_distance(coords)   

    coords['distance_invaded'] = fn.get_euclidian_distance_toCoordinates(coords, coords_patch_to_invade['x'], coords_patch_to_invade['y'])
    coords['distance_center'] = fn.get_euclidian_distance_toCoordinates(coords, np.mean(extentx), np.mean(extentx))
    coords['degree_patch'] = np.sum(dist, axis = 0) + np.sum(dist, axis = 1)
    coords['landscape_seed'] = seed_index
    
    landscape_characteristics = pd.concat([landscape_characteristics, coords])

    list_patches_to_restore = [coords[coords['position'] == 'center']['Patch'].tolist()]
    list_restoration_types = ['clustered']  
    list_seeds = [None]
    for seed in np.arange(5): ## draw 5 random patches to improve across the landscape
        np.random.seed(seed)
        list_patches_to_restore = list_patches_to_restore + [np.random.choice(np.arange(15), 5, replace=False)] # random number between [0,15)
        list_restoration_types = list_restoration_types + ['scattered'] 
        list_seeds = list_seeds + [seed]

## list of the different restoration types and order in which patches are restored        
restored_patches_dtf = pd.DataFrame({'restoration_seed': list_seeds, 
                   'restoration_types':list_restoration_types,
                   'patches_to_restore':list_patches_to_restore})
    

## characteristics of the landscape
    
## map of scattered vs clustered improvements
restoration_characteristics = pd.DataFrame()
for seed, sequence, restoration_type in zip(restored_patches_dtf['restoration_seed'], restored_patches_dtf['patches_to_restore'], restored_patches_dtf['restoration_types']):
    
    restoration_characteristics_temp = landscape_characteristics.copy()
    restoration_characteristics_temp['restoration_seed'] = seed
    restoration_characteristics_temp['restoration_type'] = restoration_type
    restoration_characteristics_temp['restored'] = np.isin(restoration_characteristics_temp['Patch'], sequence)
    
    restoration_characteristics = pd.concat([restoration_characteristics, restoration_characteristics_temp])



# %%% Map of landscapes - Figure 1

map_landscape = restoration_characteristics[restoration_characteristics['landscape_seed'] == 3]

distance_left = np.array(((map_landscape['x'] - extentx[0])**2 + (map_landscape['y'] - extentx[1])**2)**(1/2))
patch_to_invade = np.argmin(distance_left)
coords_patch_to_invade = tuple(map_landscape[(map_landscape['Patch'] == patch_to_invade) &
                                       (map_landscape['restoration_type'] == 'clustered')][['x','y']].iloc[0] - 0.01)


fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
plt.tight_layout(pad = 3)

sb.scatterplot(data = map_landscape[map_landscape['restoration_type'] == 'clustered'], 
               x = 'x',y = 'y', hue = 'restored', palette = 'coolwarm',
               ax = ax1)
ax1.annotate('', xy=coords_patch_to_invade, xytext=(0,0.35),
             arrowprops=dict(facecolor='black', shrink=0.01))
ax1.set_title('Clustered')
ax1.set_ylim(-0.01,0.41)
ax1.set_xlim(-0.01,0.41)
ax1.set_ylabel('')
ax1.set_xlabel('')
ax1.legend().remove()

sb.scatterplot(data = map_landscape[(map_landscape['restoration_type'] == 'scattered') & 
                                    (map_landscape['restoration_seed'] == 1)], 
               x = 'x',y = 'y', hue = 'restored', palette = 'coolwarm',
               ax = ax2)
ax2.annotate('', xy=coords_patch_to_invade, xytext=(0,0.35),
             arrowprops=dict(facecolor='black', shrink=0.01))
ax2.set_title('Scattered')
ax2.set_ylim(-0.01,0.41)
ax2.set_xlim(-0.01,0.41)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.legend().remove()

plt.savefig('D:/TheseSwansea/SFT/Figures/Map-ClusteredVsScattered.png', dpi = 400, bbox_inches = 'tight')



# %% plotting dynamics

# Script to visualise population dynamics

#%%%% Initial dynamics


os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3')
init_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Homogeneous/seed3/'+i for i in os.listdir() if '.pkl' in i and 'InitialPopDynamics' in i]


for f in init_15P_files[0]:
    
    print(f)
    
    # load file with population dynamics
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses

    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']
    
    for p in range(3):
   
        ind = p + 1
                
        plt.subplot(1, 3, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
                
    plt.title("Invasion - control")
    
    plt.savefig(f'D:/TheseSwansea/Patch-Models/Figures/PopDynamics.png', dpi = 400, bbox_inches = 'tight')

    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10)])
    plt.show()



#%%%% Controls

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in os.listdir() if 'invasion' not in i and ('.pkl' in i or '.npy' in i)]


for f in control_15P_files:
    
    print(f)
    
    # load file with population dynamics
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses

    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']
    
    for p in range(P):
   
        ind = p + 1
                
        plt.subplot(3, 5, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)])
                
    plt.title("Invasion - control")
    plt.show()
    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10)])
    plt.show()





#%%%% Controls - invasion  - corner patch

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3')
control_invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Control/seed3/'+i for i in 
                              os.listdir() if '.pkl' in i and 'invasion' in i and '-10000-' not in i and 'CornerPatch' in i]


for f in control_invasion_15P_files:
    
    print(f)
    
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses
    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']


    for p in range(P):
   
        ind = p + 1
        
        invaders = FW_new['invaders'][p]
        
        plt.subplot(3, 5, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,~invaders])
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                   solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,invaders], linestyle="dotted")

        
    plt.title("Invasion - control")
    plt.show()
    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,~FW_new['invaders'].reshape(P*Stot)])
    plt.title(f'{f}')
    plt.show()
    
    # species who invaded middle patches 
    # species who had zero biomass (FW_new['y0']) in centre patches before invasion
    solY_centre_patches = solY[:,np.repeat(FW_new['coords']['position'] == 'center',Stot)]
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY_centre_patches[np.arange(0,solT.shape[0],10),:][:,(sol['FW_new']['y0'][FW_new['coords']['position'] == 'center'].reshape(5*Stot) == 0)])
    plt.title(f'invaders into middle patches')
    plt.show()
    
    # species who weren't initialised in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) == 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dotted')
    plt.title(f'y0 == 0 in initial period')
    plt.show()
    
    # species who went extinct in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) != 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dashed')
    plt.title(f'went extinct in initial period')
    plt.show()
    
    '''
    
    It seems that even among the species that were initialised and went extinct during the initial run
    species can still invade once dynamics have stabilised, in the absence of restoration
    
    '''
    
    
    
#%%%% invasion - restoration - corner patch

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow/')
invasion_15P_files = ['D:/TheseSwansea/Patch-Models/outputs/15Patches/Heterogeneous/seed3/invasion/CornerPatch/narrow/'+i for i in 
                              os.listdir() if '.pkl' in i and 'invasion' in i]


for f in invasion_15P_files:
    
    print(f)
    
    if '.npy' in f:
        sol = np.load(f, allow_pickle=True).item()
    elif ".pkl" in f:
        with open(f, 'rb') as file: 
            sol = pickle.load(file)  
        file.close()
    
    solT = sol['t'] ## time
    solY = sol['y'] ## biomasses
    
    FW_new = sol['FW_new'] ## subset food web (this goes with sp_ID_)
    Stot_new = FW_new['Stot']


    for p in range(P):
   
        ind = p + 1
        
        invaders = FW_new['invaders'][p]
        
        plt.subplot(3, 5, ind)
        plt.tight_layout()
        
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                    solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,~invaders])
        plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                   solY[np.arange(0,solT.shape[0],10),:][:,range(Stot_new*ind-Stot_new,Stot_new*ind)][:,invaders], linestyle="dotted")

        
    plt.title(f'')
    plt.show()
    
    ## plotting all dynamics together
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,~FW_new['invaders'].reshape(P*Stot)])
    plt.title(f'{f}')
    plt.show()
    
    # species who invaded middle patches 
    # species who had zero biomass (FW_new['y0']) in centre patches before invasion
    solY_centre_patches = solY[:,np.repeat(FW_new['coords']['position'] == 'center',Stot)]
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY_centre_patches[np.arange(0,solT.shape[0],10),:][:,(sol['FW_new']['y0'][FW_new['coords']['position'] == 'center'].reshape(5*Stot) == 0)])
    plt.title(f'invaders into middle patches')
    plt.show()
    
    # species who weren't initialised in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) == 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dotted')
    plt.title(f'y0 == 0 in initial period')
    plt.show()
    
    # species who went extinct in the initial period
    plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
                solY[np.arange(0,solT.shape[0],10),:][:,(sol['FW']['y0'].reshape(P*Stot) != 0) & (FW_new['invaders'].reshape(P*Stot))], linestyle='dashed')
    plt.title(f'went extinct in initial period')
    plt.show()
    
    '''
    
    It seems that even among the species that were initialised and went extinct during the initial run
    species can still invade once dynamics have stabilised, in the absence of restoration
    
    '''




# %% Load datasets

P = 15
## 
os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')

# 'res15...' is the dataframe with species-level characteristics such as biomass, species traits etc
# FW15... is the dataframe with local (patch-level) characteristics such as species richness, mfcl, modularity etc
# FW15_chara... is the dataframe with regional (landscape level) characteristics such as regional species richness, etc


# %%% Initial population dynamics, before restoration


## load in initial characteristics

# species-level
res15_init_normal = pd.read_csv(f'ResultsInitial-narrow_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
res15_init_normal['stage'] = 'init' ## add a 'stage' column
# patch-level
FW15_init_normal = pd.read_csv(f'ResultsInitial-narrow-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
FW15_init_normal['stage'] = 'init'

## add suffixes '_init' to initial columns - for merging later with restored and recolonised dataframes
cols = ['file', 'type', 'quality_ratio',
       'nb_improved', 'S_local', 'C_local', 'L_local', 'LS_local',
       'MeanBiomass_local', 'Modularity_local', 'MeanGen_local',
       'MeanVul_local', 'StdGen_local', 'StdVul_local', 'MeanTL_local',
       'MeanTP_local', 'Mfcl_local', 'MeanBodyMass_local', 'nb_top_local',
       'nb_int_local', 'nb_herb_local', 'nb_plants_local',
       'gamma_diversity_shannon', 'mean_alpha_diversity_shannon',
       'beta_diversity_shannon', 'alpha_diversity_shannon', 'deltaR',
       'simulation_length', 'simulation_length_years', 'stage']
FW15_init_normal = FW15_init_normal.rename(columns={c: c+'_init' for c in FW15_init_normal.columns if c in cols})

## add suffixes '_init' to initial columns - for merging later with restored and recolonised dataframes
cols = ['file', 'Stot', 'Stot_new', 'type', 'quality_ratio',
       'nb_improved', 
       'mean_TL_extinct', 'mean_TP_extinct', 'mean_C_extinct',
       'mean_Vul_extinct', 'mean_Gen_extinct', 'nb_extinct',
       'extinct_any_patch', 'extinct', 'gamma_diversity',
       'mean_alpha_diversity', 'beta_diversity', 'deltaR', 'B_final', 'B_init',
       'TL', 'TP', 'C', 'Vul', 'Gen', 'BS', 'stage']
res15_init_normal = res15_init_normal.rename(columns={c: c+'_init' for c in res15_init_normal.columns if c in cols})


## initial landscape characteristics
FW15_chara_init = pd.read_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/ResultsInitial-narrow-RegionalFoodWebChara_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
## add suffix '_landscape_init' to columns 
cols = ['type', 'quality_ratio', 'MeanBodyMass', 'L', 'C', 'LS', 'MeanInteractionStrength',
'StdInteractionStrength', 'StdVul', 'StdGen', 'MeanVul', 'MeanGen',
'Modularity', 'Mfcl', 'S', 'S_top', 'S_int', 'S_herb', 'S_plants',
'S_init', 'S_top_init', 'S_int_init', 'S_herb_init', 'S_plants_init']
FW15_chara_init = FW15_chara_init.rename(columns={c: c+'_landscape_init' for c in FW15_chara_init.columns if c in cols})
# rename file as file_init and not file_landscape_init so that it merges properly below
FW15_chara_init = FW15_chara_init.rename(columns={'file':'file_init'})


## merge initial landscape (regional) characteristics and initial pop dynamics
FW15_init_normal = pd.merge(FW15_init_normal, FW15_chara_init, on = ['landscape_seed', 'file_init', 'sim'])


## looking at whether the random initialisation of species across different food webs changes
## their initial species composition

## proportion of species across trophic levels as they are randomly seeded into the landscape
prop_init_TL = res15_init_normal[res15_init_normal['B_init_init'] > 0].groupby(['sim','TL_init']).agg({'TL_init': ['size']})
prop_init_TL.columns = ['init_TL_count']
prop_init_TL['init_TL_count_total'] = prop_init_TL.groupby(level=0)['init_TL_count'].transform('sum')
prop_init_TL['init_TL_percentage'] = (prop_init_TL['init_TL_count'] / prop_init_TL['init_TL_count_total']) * 100


# after initial run:
    ## proportion of species across trophic levels after initial population dynamics

prop_init1_TL = res15_init_normal[res15_init_normal['B_final_init'] > 0].groupby(['sim','TL_init']).agg({'TL_init': ['size']})
prop_init1_TL.columns = ['init_TL_count']
prop_init1_TL['init_TL_count_total'] = prop_init1_TL.groupby(level=0)['init_TL_count'].transform('sum')
prop_init1_TL['init_TL_percentage'] = (prop_init1_TL['init_TL_count'] / prop_init1_TL['init_TL_count_total']) * 100


# let's look at the results
sb.scatterplot(pd.merge(prop_init1_TL, prop_init_TL, on = ['sim', 'TL_init']), 
               y = 'init_TL_percentage_y',
               x = 'init_TL_percentage_x', 
               hue = 'TL_init',
               palette = palette_colors)

## it doesn't seem like the initial seeding across different simulations or TL particularly affects 
## the final percentages - apart maybe for TL3 where there is a small linear relationship


# %%% put everything together

os.chdir('D:/TheseSwansea/Patch-Models/outputs/15Patches')

## restored population dynamics at the species level (res) and at the patch level (FW)
res15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
FW15_invasion = pd.read_csv(f'ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch-FoodwebMetrics_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
res15_invasion['landscape'] = 'normal'
FW15_invasion['landscape'] = 'normal'
res15_invasion['stage'] = 'restored'
FW15_invasion['stage'] = 'restored'

## control population dynamics at the species level (res) and at the patch level (FW)
res15_control_invasion_normal = pd.read_csv(f'Control-Invasion-seed3-narrow-CornerPatch-handmade_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
FW15_control_invasion_normal = pd.read_csv(f'Control-Invasion-seed3-narrow-CornerPatch-FoodwebMetrics-handmade_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
res15_control_invasion_normal['landscape'] = 'normal'
FW15_control_invasion_normal['landscape'] = 'normal'
res15_control_invasion_normal['stage'] = 'control'
FW15_control_invasion_normal['stage'] = 'control'

### for check below
test1 = FW15_control_invasion_normal.groupby(['sim','landscape_seed']).agg({'S_local':['mean','count', 'std']})

# FW15_invasion['restored_patches_seed'] = np.nan_to_num(FW15_invasion['restored_patches_seed'], nan=-1)

# we expand the control dataset to match the number of restorations that were
# implemented, i.e. 5 restoration seeds (scattered) + clustered
# because there is only one control per food web replicate x landscape seed (no replicated for restoration sequence as this is pre-restoration)
nrow = FW15_control_invasion_normal.shape[0]
FW15_control_invasion_expanded = pd.concat([FW15_control_invasion_normal]*
                                         len(np.unique(FW15_invasion['restored_patches_seed']))) # repeat copies of the control dataframe
FW15_control_invasion_expanded['restored_patches_seed'] = np.repeat(np.unique(FW15_invasion['restored_patches_seed']), nrow)
FW15_control_invasion_expanded['restoration_type'] = ['clustered' if np.isnan(i) else 'scattered' for i in FW15_control_invasion_expanded['restored_patches_seed']]

nrow = res15_control_invasion_normal.shape[0]
res15_control_invasion_normal = pd.concat([res15_control_invasion_normal]*
                                         len(np.unique(res15_invasion['restored_patches_seed']))) # repeat copies of the control dataframe
res15_control_invasion_normal['restored_patches_seed'] = np.repeat(np.unique(res15_invasion['restored_patches_seed']), nrow)
res15_control_invasion_normal['restoration_type'] = ['clustered' if np.isnan(i) else 'scattered' for i in res15_control_invasion_normal['restored_patches_seed']]


## CHECKS
test = res15_control_invasion_normal.groupby(['sim','restoration_type','nb_improved','landscape_seed','restored_patches_seed'], dropna = False).agg({'B_final':['mean','count', 'std']}).reset_index()
test = FW15_control_invasion_expanded.groupby(['sim','restoration_type','nb_improved','landscape_seed','restored_patches_seed'], dropna = False).agg({'S_local':['mean','count', 'std']}).reset_index()
## values should match test1
## missing 'clustered' type bc its restored seed is 'nan'


## add control (nb_improved == 0) to restored dataframes
res15_invasion_normal = pd.concat([res15_control_invasion_normal, res15_invasion])
FW15_invasion_normal = pd.concat([FW15_control_invasion_expanded, FW15_invasion])


## adding info about food webs pre-invasion (with suffixe '_init') 
res15_invasion_normal = pd.merge(res15_invasion_normal, res15_init_normal, 
                                 on = ['landscape_seed','patch','sim','sp_ID'], 
                                 suffixes = ['', '_init'])  
## add patch level + regional level initial info
res15_invasion_normal = pd.merge(res15_invasion_normal, FW15_init_normal, 
                                 on = ['landscape_seed','patch','sim'], 
                                 suffixes = ['', '_init'])


## landscape characteristics for restored and control runs
FW15_chara = pd.read_csv(f'ResultsHeterogeneous-seed3-narrow-invasion-CornerPatch-RegionalFoodWebChara_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])
FW15_chara_control = pd.read_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/Control-Invasion-seed3-narrow-invasion-CornerPatch-RegionalFoodWebChara_sim_{P}Patches_{Stot}sp_{C}C_24092024.csv', index_col=[0])

# expand control variables so that there is one per restoration sequence 
nrow = FW15_chara_control.shape[0]
FW15_chara_control = pd.concat([FW15_chara_control]*
                                         len(np.unique(FW15_chara['restored_patches_seed']))) # repeat copies of the control dataframe
FW15_chara_control['restored_patches_seed'] = np.repeat(np.unique(FW15_chara['restored_patches_seed']), nrow)
FW15_chara_control['restoration_type'] = ['clustered' if np.isnan(i) else 'scattered' for i in FW15_chara_control['restored_patches_seed']]

# combine control and restored landscape characteristics
FW15_chara = pd.concat([FW15_chara, FW15_chara_control])

## adding info about food webs after invasion
FW15_invasion_normal = pd.merge(FW15_invasion_normal, FW15_chara, on = 
                                ['landscape_seed', 'file', 'sim', 'type', 'quality_ratio',
                                 'restoration_type', 'restored_patches_seed', 'nb_improved',
        'latest_patch_improved', 'S_regional'], how = 'outer')

## adding info about food webs pre-invasion (with suffixe '_init)
FW15_invasion_normal = pd.merge(FW15_invasion_normal, FW15_init_normal, 
                                 on = ['landscape_seed','patch','sim'], 
                                 suffixes = ['', '_init'])


## calculate persistence (proportion of extant species compared to regional food web)
FW15_init_normal['persistence'] = FW15_init_normal['S_local_init']/FW15_init_normal['S_init_landscape_init']
FW15_init_normal['persistence_plants'] = FW15_init_normal['nb_plants_local_init']/FW15_init_normal['S_plants_init_landscape_init']
FW15_init_normal['persistence_herb'] = FW15_init_normal['nb_herb_local_init']/FW15_init_normal['S_herb_init_landscape_init']
FW15_init_normal['persistence_int'] = FW15_init_normal['nb_int_local_init']/FW15_init_normal['S_int_init_landscape_init']
FW15_init_normal['persistence_top'] = FW15_init_normal['nb_top_local_init']/FW15_init_normal['S_top_init_landscape_init']

FW15_invasion_normal['persistence'] = FW15_invasion_normal['S_local']/FW15_invasion_normal['S_init_landscape_init']
FW15_invasion_normal['persistence_plants'] = FW15_invasion_normal['nb_plants']/FW15_invasion_normal['S_plants_init_landscape_init']
FW15_invasion_normal['persistence_herb'] = FW15_invasion_normal['nb_herb']/FW15_invasion_normal['S_herb_init_landscape_init']
FW15_invasion_normal['persistence_int'] = FW15_invasion_normal['nb_int']/FW15_invasion_normal['S_int_init_landscape_init']
FW15_invasion_normal['persistence_top'] = FW15_invasion_normal['nb_top']/FW15_invasion_normal['S_top_init_landscape_init']


## subset simulations that ran completely (5 food webs x 5 landscapes)
list_sim = [0,1,2,6,13] ## 1,2,6,13 keep only simulations that ran all the way # 9 sim has a two simulations that did not run
list_landscapes = [0,1,2,3,4] ## remove landscape 5 whcih had two identical patches

res15_invasion_normal.shape
res15_invasion_normal = res15_invasion_normal[np.isin(res15_invasion_normal['sim'], list_sim) &
                                              np.isin(res15_invasion_normal['landscape_seed'], 
                                                      list_landscapes)]
res15_invasion_normal.shape

FW15_invasion_normal = FW15_invasion_normal[np.isin(FW15_invasion_normal['sim'], list_sim) &
                                            np.isin(FW15_invasion_normal['landscape_seed'], list_landscapes)]

FW15_init_normal = FW15_init_normal[np.isin(FW15_init_normal['sim'], list_sim) &
                                            np.isin(FW15_init_normal['landscape_seed'], list_landscapes)]

res15_init_normal = res15_init_normal[np.isin(res15_init_normal['sim'], list_sim) &
                                            np.isin(res15_init_normal['landscape_seed'], list_landscapes)]

FW15_control_invasion_normal = FW15_control_invasion_normal[np.isin(FW15_control_invasion_normal['sim'], list_sim) &
                                            np.isin(FW15_control_invasion_normal['landscape_seed'], 
                                                    list_landscapes)]


## to test what order patches improvement matters, we calculate by how much the improvement of a patch 
## increased a metric


test = res15_invasion_normal.groupby(['sim','restoration_type','nb_improved','landscape_seed'], dropna = False).agg({'B_final':['mean','count', 'std'], 'restored_patches_seed':'unique'})
res15_invasion_normal['sim'].unique()

FW15_invasion_normal.groupby(['restoration_type'], dropna = False).agg({'S_local':'size'})


# %% stats about initial communities

FW15_init_normal['S_local_init'].describe()
FW15_init_normal['LS_local_init'].describe()
FW15_init_normal['Mfcl_local_init'].describe()

FW15_control_invasion_normal['S_local'].describe()
FW15_control_invasion_normal['LS_local'].describe()
FW15_control_invasion_normal['Mfcl_local'].describe()


## compare initial species richness with different levels of improvent using Mann Whitney U test (nonparametric statistical test)
from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(FW15_init_normal['S_local_init'], 
                         FW15_control_invasion_normal['S_local'])
stat, p_value

stat, p_value = mannwhitneyu(FW15_init_normal['S_local_init'], 
                         FW15_invasion_normal[FW15_invasion_normal['nb_improved'] == 1]['S_local'])
stat, p_value

sb.histplot(FW15_init_normal['S_local_init'], bins = 50)
sb.histplot(FW15_control_invasion_normal['S_local'], bins = 50)


## biomass
res15_init_normal['B_final_init'].describe()
res15_invasion_normal[(res15_invasion_normal['nb_improved'] == 1) &
                      (res15_invasion_normal['B_final'] > 0)]['B_final'].describe()

sb.histplot(np.log(res15_init_normal[res15_init_normal['B_final_init'] > 0]['B_final_init']), bins = 50)
sb.histplot(np.log(res15_invasion_normal[res15_invasion_normal['nb_improved'] == 1]['B_final']), bins = 50)

from scipy.stats import shapiro
stat, p_value = shapiro(np.log(res15_invasion_normal[(res15_invasion_normal['nb_improved'] == 1) &
                      (res15_invasion_normal['B_final'] > 0)]['B_final']))

stat, p_value = mannwhitneyu(res15_init_normal[(res15_init_normal['B_final_init'] > 0) &
                                               (res15_init_normal['TL_init']  == 0)]['B_final_init'], 
                         res15_invasion_normal[(res15_invasion_normal['nb_improved'] == 5) &
                                               (res15_invasion_normal['B_final'] > 0) & 
                                               (res15_invasion_normal['TL'] == 0)]['B_final'])
stat, p_value


res15_init_normal[(res15_init_normal['B_final_init'] > 0) & 
                  (res15_init_normal['TL_init']  == 2)]['B_final_init'].describe()
res15_invasion_normal[(res15_invasion_normal['nb_improved'] == 5) &
                      (res15_invasion_normal['B_final'] > 0) & 
                      (res15_invasion_normal['TL'] == 2)]['B_final'].describe()


## persistence

FW15_init_normal['persistence'].describe()
FW15_invasion_normal.groupby(['nb_improved']).agg({
    'persistence':'mean'})


# %% species area curve

# Perform the grouping and calculate the cumulative sum over 'S_local'
FW15_invasion_normal['cumulative_S_local'] = FW15_invasion_normal[FW15_invasion_normal['stage'] != 'init'].groupby([
    'sim','restoration_type','landscape_seed','restored_patches_seed',
    'nb_improved','latest_patch_improved'], dropna=False)['S_local'].cumsum()

# View the updated DataFrame
print(FW15_invasion_normal.head())




# Group by the relevant columns including 'patch' (or a similar column indicating the patch order)
grouped = res15_init_normal.groupby([
    'sim','landscape_seed'], dropna=False, as_index = False)


grouped.agg({'sp_ID':'size'})

# Define a function to calculate the cumulative sum of new unique 'Sp_ID'
def cumulative_new_unique_sp_id(group):
    
    # Track the unique 'Sp_ID' encountered so far
    encountered_sp_ids = np.array([])
    cumulative_sum = []
        
    for patch in group['patch'].unique():
        
        sub = group[group['patch'] == patch]
        
        # Current set of unique 'Sp_ID' in this patch
        current_sp_ids = sub[sub['B_final_init'] > 0]['sp_ID'].unique()
        
        # Update the encountered 'Sp_ID' set
        encountered_sp_ids = np.concatenate([encountered_sp_ids,current_sp_ids])
        # print(encountered_sp_ids)
        
        cumulative_sum_value = len(np.unique(encountered_sp_ids))
        # Return the cumulative sum as a new column
        group.loc[group['patch'] == patch, 'cumulative_new_sp_id'] = cumulative_sum_value
        
    return group


# Apply the function to each group
result = grouped.apply(cumulative_new_unique_sp_id)
result = result.groupby(['sim','landscape_seed','patch']).agg({'cumulative_new_sp_id': lambda x: np.unique(x)[0]})

sb.pointplot(data = result, y = 'cumulative_new_sp_id', x = 'patch')
plt.xlabel('Number of patches')
plt.ylabel('Cumlative number of species')


## this is the species area curve taking into account order of restoration

def cumulative_new_unique_sp_id_OrderRestoration(group):
    
    # Track the unique 'Sp_ID' encountered so far
    
    result = pd.DataFrame()
    ind = 0   # index of dataframe 
    for index, row in restored_patches_dtf.iterrows():
        
        sequence = row['patches_to_restore']
        restoration_seed = row['restoration_seed']
        
        count = 0
        encountered_sp_ids = encountered_top_sp_ids = encountered_int_sp_ids = encountered_herb_sp_ids = encountered_plants_sp_ids = np.array([])
        cumulative_biomass = cumulative_biomass_top = cumulative_biomass_int = cumulative_biomass_herb = cumulative_biomass_plants = np.nan
        for patch in sequence:
            
            ind+=1
            count+=1
            sub = group[(group['patch'] == patch)]
                        
            # Current set of unique 'Sp_ID' in this patch
            current_sp_ids = sub[sub['B_final_init'] > 0]['sp_ID'].unique()
            current_top_sp_ids = sub[(sub['B_final_init'] > 0) & (sub['TL_init'] == 3)]['sp_ID'].unique()
            current_int_sp_ids = sub[(sub['B_final_init'] > 0) & (sub['TL_init'] == 2)]['sp_ID'].unique()
            current_herb_sp_ids = sub[(sub['B_final_init'] > 0) & (sub['TL_init'] == 1)]['sp_ID'].unique()
            current_plants_sp_ids = sub[(sub['B_final_init'] > 0) & (sub['TL_init'] == 0)]['sp_ID'].unique()
            
            # Update the encountered 'Sp_ID' set
            encountered_sp_ids = np.concatenate([encountered_sp_ids,current_sp_ids])
            encountered_top_sp_ids = np.concatenate([encountered_top_sp_ids,current_top_sp_ids])
            encountered_int_sp_ids = np.concatenate([encountered_int_sp_ids,current_int_sp_ids])
            encountered_herb_sp_ids = np.concatenate([encountered_herb_sp_ids,current_herb_sp_ids])
            encountered_plants_sp_ids = np.concatenate([encountered_plants_sp_ids,current_plants_sp_ids])
            
            cumulative_sum_value = len(np.unique(encountered_sp_ids))
            cumulative_sum_value_top = len(np.unique(encountered_top_sp_ids))
            cumulative_sum_value_int = len(np.unique(encountered_int_sp_ids))
            cumulative_sum_value_herb = len(np.unique(encountered_herb_sp_ids))
            cumulative_sum_value_plants = len(np.unique(encountered_plants_sp_ids))
            # Return the cumulative sum as a new column
            
            # np.stack allows for zero dimensional arrays to be concatenated
            
            cumulative_biomass = np.nanmean(np.stack([cumulative_biomass, np.mean(sub[sub['B_final_init'] > 0]['B_final_init'])]))
            cumulative_biomass_top = np.nanmean(np.stack([cumulative_biomass_top, np.mean(sub['B_final_init'][(sub['B_final_init'] > 0) & (sub['TL_init'] == 3)])]))
            cumulative_biomass_int = np.nanmean(np.stack([cumulative_biomass_int, np.mean(sub['B_final_init'][(sub['B_final_init'] > 0) & (sub['TL_init'] == 2)])]))
            cumulative_biomass_herb = np.nanmean(np.stack([cumulative_biomass_herb, np.mean(sub['B_final_init'][(sub['B_final_init'] > 0) & (sub['TL_init'] == 1)])]))
            cumulative_biomass_plants = np.nanmean(np.stack([cumulative_biomass_plants, np.mean(sub['B_final_init'][(sub['B_final_init'] > 0) & (sub['TL_init'] == 0)])]))

            result = pd.concat([result,
                                pd.DataFrame({'patch':patch,
                                              'restored_patch_seed': restoration_seed,
                                              
                                              'cumulative_new_sp_id_all': cumulative_sum_value,
                                              'cumulative_new_sp_id_top': cumulative_sum_value_top,
                                              'cumulative_new_sp_id_int':cumulative_sum_value_int,
                                              'cumulative_new_sp_id_herb': cumulative_sum_value_herb,
                                              'cumulative_new_sp_id_plants': cumulative_sum_value_plants,
                                              
                                              'cumulative_biomass_all':cumulative_biomass,
                                              'cumulative_biomass_top':cumulative_biomass_top,
                                              'cumulative_biomass_int':cumulative_biomass_int,
                                              'cumulative_biomass_herb':cumulative_biomass_herb,
                                              'cumulative_biomass_plants':cumulative_biomass_plants,
                                              
                                              'latest_patch_improved': patch,
                                              'count_patch_improved': count,
                                              'landscape_seed':sub['landscape_seed'].unique().tolist(),
                                              'sim':sub['sim'].unique().tolist()})
                                ])

    return result

cumulative_SAC = grouped.apply(cumulative_new_unique_sp_id_OrderRestoration)
cumulative_SAC = cumulative_SAC.reset_index()

## 1 versus 5 patches
sb.histplot(cumulative_SAC[cumulative_SAC['count_patch_improved'] == 1]['cumulative_new_sp_id_all'], bins = 50)
sb.histplot(cumulative_SAC[cumulative_SAC['count_patch_improved'] == 5]['cumulative_new_sp_id_all'], bins = 50)

# clustered versus scattered
sb.histplot(cumulative_SAC[(~np.isnan(cumulative_SAC['restored_patch_seed'])) &
                           (cumulative_SAC['count_patch_improved'] == 1)]['cumulative_new_sp_id_all'], bins = 50)
sb.histplot(cumulative_SAC[(np.isnan(cumulative_SAC['restored_patch_seed'])) &
                           (cumulative_SAC['count_patch_improved'] == 1)]['cumulative_new_sp_id_all'], bins = 50)

sb.histplot(cumulative_SAC[(~np.isnan(cumulative_SAC['restored_patch_seed'])) &
                           (cumulative_SAC['count_patch_improved'] == 5)]['cumulative_new_sp_id_all'], bins = 50)
sb.histplot(cumulative_SAC[(np.isnan(cumulative_SAC['restored_patch_seed'])) &
                           (cumulative_SAC['count_patch_improved'] == 5)]['cumulative_new_sp_id_all'], bins = 50)



## compare number of species gained between scattered and clustered scenarios
group1 = cumulative_SAC[(~np.isnan(cumulative_SAC['restored_patch_seed'])) &
                           (cumulative_SAC['count_patch_improved'] == 1)]['cumulative_new_sp_id_all']
group2 = cumulative_SAC[(np.isnan(cumulative_SAC['restored_patch_seed'])) &
                           (cumulative_SAC['count_patch_improved'] == 1)]['cumulative_new_sp_id_all']

group1.describe()
group2.describe()

from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(group1, group2)
stat, p_value

FW15_invasion_normal.columns

cumulative_SAC_plot = pd.merge(cumulative_SAC, landscape_characteristics, 
                               left_on = ['landscape_seed', 'latest_patch_improved'],
                               right_on = ['landscape_seed', 'Patch'])
sb.lmplot(cumulative_SAC_plot[(cumulative_SAC_plot['count_patch_improved'] == 1)], 
               y = 'cumulative_new_sp_id_all', x = 'degree_patch')








landscape_change = FW15_invasion_normal.groupby([
    'sim','restoration_type','landscape_seed','restored_patches_seed','patch',
    'nb_improved','latest_patch_improved'], dropna = False).agg({
        'S_local':['mean','count']})
landscape_change.columns = list(map(''.join, landscape_change.columns.values))

diff_dataframe = landscape_change.groupby(level=[0,1,2,3,4], dropna = False).agg({
        'S_localmean':['pct_change', 'diff'],
        }).reset_index()        

diff_dataframe.columns = list(map(''.join, diff_dataframe.columns.values))

diff_dataframe = pd.merge(diff_dataframe, landscape_characteristics, 
                left_on = ['landscape_seed', 'latest_patch_improved'],
                right_on = ['landscape_seed', 'Patch'], 
                how = 'outer')



biomass_change = res15_invasion_normal.groupby([
    'sim','restoration_type','landscape_seed','restored_patches_seed','patch',
    'nb_improved','latest_patch_improved'], dropna = False).agg({
        'B_final':['mean','count']})
biomass_change.columns = list(map(''.join, biomass_change.columns.values))

biomass_diff_dataframe = biomass_change.groupby(level=[0,1,2,3,4], dropna = False).agg({
        'B_finalmean':['pct_change', 'diff'],
        }).reset_index()        

biomass_diff_dataframe.columns = list(map(''.join, biomass_diff_dataframe.columns.values))

biomass_diff_dataframe = pd.merge(biomass_diff_dataframe, landscape_characteristics, 
                left_on = ['landscape_seed', 'latest_patch_improved'],
                right_on = ['landscape_seed', 'Patch'], 
                how = 'outer')


# %%%% plot of effect of first patch improvement on species richness


fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex = True)
fig.tight_layout(pad = 3)

sb.regplot(cumulative_SAC_plot[(cumulative_SAC_plot['count_patch_improved'] == 1)], 
               y = 'cumulative_biomass_all', x = 'degree_patch', 
               ax = ax1)
ax1.set_ylabel('Initial biomass')
ax1.set_xlabel('Mean euclidian distance of first'
               '\n'
               'improved patch to other patches')
ax1.set_title('Initial biomass in first'
              '\n'
              'restored patch (before restoration)')

sb.regplot(diff_dataframe[diff_dataframe['nb_improved'] == 1], 
               x = 'degree_patch', y = 'S_localmeandiff', ax = ax2)
ax2.set_ylabel('Number of species gained per patch')
ax2.set_xlabel('Mean euclidian distance of'
               '\n'
               'improved patch to other patches')
ax2.set_title('Species gained in any patch depending on the'
              '\n'
              'location of the first patch improved')

plt.savefig('D:/TheseSwansea/SFT/Figures/FirstPatchImprovement.png', dpi = 400, bbox_inches = 'tight')


# %%%% plot of effect of first patch improvement on biomass


fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex = True)
fig.tight_layout(pad = 3)

sb.regplot(cumulative_SAC_plot[(cumulative_SAC_plot['count_patch_improved'] == 1)], 
               y = 'cumulative_new_sp_id_all', x = 'degree_patch', 
               ax = ax1)
ax1.set_ylabel('Initial species richness')
ax1.set_xlabel('Mean euclidian distance of first'
               '\n'
               'improved patch to other patches')
ax1.set_title('Initial species richness in first'
              '\n'
              'restored patch (before restoration)')

sb.regplot(biomass_diff_dataframe[biomass_diff_dataframe['nb_improved'] == 1], 
               x = 'degree_patch', y = 'B_finalmeandiff', ax = ax2)
ax2.set_ylabel('Number of species gained per patch')
ax2.set_xlabel('Mean euclidian distance of first'
               '\n'
               'improved patch to other patches')
ax2.set_title('Species gained in any patch'
              '\n'
              'after one patch improvement')

plt.savefig('D:/TheseSwansea/SFT/Figures/FirstPatchImprovement.png', dpi = 400, bbox_inches = 'tight')



# %%%% plots of initial SAC versus recolonisation outcomes

fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex = True)
fig.tight_layout(pad = 3)

sb.pointplot(data = cumulative_SAC[np.isnan(cumulative_SAC['restored_patch_seed'])], 
             y = 'cumulative_new_sp_id_all', x = 'count_patch_improved', 
             color = 'black', ax = ax1, order = [0,1,2,3,4,5])
sb.pointplot(data = cumulative_SAC[~np.isnan(cumulative_SAC['restored_patch_seed'])], y = 'cumulative_new_sp_id_all', x = 'count_patch_improved',
             linestyles='dotted', color = 'grey', hue = 'restored_patch_seed', palette = 'coolwarm',
             ax = ax1)
ax1.set_xlabel('Number of patches')
ax1.set_ylabel('Cumulative number of species')
ax1.legend(title = 'Restoration sequence', bbox_to_anchor = [2,1.3], ncol = 6)


sb.pointplot(data = FW15_invasion_normal[np.isnan(FW15_invasion_normal['restored_patches_seed'])], 
             y = 'S_local', x = 'nb_improved', 
             color = 'black', ax = ax2)
sb.pointplot(data = FW15_invasion_normal[~np.isnan(FW15_invasion_normal['restored_patches_seed'])], 
             y = 'S_local', x = 'nb_improved', 
             linestyles='dotted', color = 'grey', hue = 'restored_patches_seed', palette = 'coolwarm', ax = ax2)
plt.xlabel('Number of patches')
plt.ylabel('Cumulative number of species')
plt.legend().remove()

plt.savefig('D:/TheseSwansea/SFT/Figures/SAC_RestorationOrder-ActualOutcome.png', dpi = 400, bbox_inches = 'tight')



# %%%% Initial cumulative SAC for clustered vs scattered and BAC across trophic levels x restoration type



fig, ([ax1, ax4]) = plt.subplots(nrows=1, ncols=2, figsize=(9,5), sharex=True)
## different trophic levels
unique_categories = ['nb_top', 'nb_int', 'nb_herb', 'nb_plants']
colors = sb.color_palette("coolwarm", len(unique_categories))


## species area relationship
# clustered
sb.pointplot(data = cumulative_SAC[np.isnan(cumulative_SAC['restored_patch_seed'])], 
             y = 'cumulative_new_sp_id_all', x = 'count_patch_improved', 
             color = 'black', ax = ax1, order = [1,2,3,4,5])
# scattered
sb.pointplot(data = cumulative_SAC[~np.isnan(cumulative_SAC['restored_patch_seed'])], 
             y = 'cumulative_new_sp_id_all', x = 'count_patch_improved',
             linestyles='dotted', color = 'grey', ax = ax1)
ax1.set_xlabel('Number of patches')
ax1.set_ylabel('Cumulative number of species' '\n' 'before recolonisation')
ax1.annotate('A', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')


biomass_columns = [
    'cumulative_biomass_plants',
    'cumulative_biomass_herb',
    'cumulative_biomass_int', 
    'cumulative_biomass_top'
]

for i, col in enumerate(biomass_columns):
    # clustered
    sb.pointplot(data=cumulative_SAC[np.isnan(cumulative_SAC['restored_patch_seed'])], 
        y=col, x='count_patch_improved', 
        color=colors[i], ax=ax4)
    #scattered
    sb.pointplot(data = cumulative_SAC[~np.isnan(cumulative_SAC['restored_patch_seed'])], 
                 y = col, x = 'count_patch_improved',
                 linestyles='dotted', 
                 color = colors[i], ax = ax4)
ax4.set_xlabel('Number of patches')
ax4.set_ylabel('Mean species biomass before recolonisation')
ax4.annotate('B', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')


plt.savefig('D:/TheseSwansea/SFT/Figures/SuppSAC.png', dpi = 400, bbox_inches = 'tight')






# %% Traits of new recolonising species across improvement levels


# Group by the relevant columns including 'patch' (or a similar column indicating the patch order)
grouped = res15_invasion_normal.groupby([
    'sim','landscape_seed','restored_patches_seed','restoration_type'], dropna=False, as_index = False)


def get_traits_latest_recolonisers(group):
    
    # Track the unique 'Sp_ID' encountered so far
    
    result = pd.DataFrame()
    ind = 0   # index of dataframe 
    
    for patch in group['patch'].unique():  
        
        restored_patch_seed = group['restored_patches_seed'].unique()[0]
        landscape_seed = group['landscape_seed'].unique()[0]
        sim = group['sim'].unique()[0]
        restoration_type = group['restoration_type'].unique()[0]
        
        encountered_sp_ids = res15_init_normal[(res15_init_normal['landscape_seed'] == landscape_seed) &
                                               (res15_init_normal['sim'] == sim) &
                                               (res15_init_normal['patch'] == patch) &
                                               (res15_init_normal['B_final_init'] > 0)]['sp_ID']
        cumulative_biomass = 0
    
        for count_improved in np.sort(group['nb_improved'].unique()):
            
        #print('patch', patch,'nb_improved', count_improved)
            
            ind+=1
            sub = group[(group['patch'] == patch) & (group['nb_improved'] == count_improved)]
                        
            # Current set of unique 'Sp_ID' in this patch
            current_sp_ids = sub[sub['B_final'] > 0]['sp_ID'].unique()
            
            # Update the encountered 'Sp_ID' set
            new_sp_ids = np.setdiff1d(current_sp_ids, encountered_sp_ids)
            # print('encountered', encountered_sp_ids, 'current', current_sp_ids)
            
            encountered_sp_ids = np.concatenate([encountered_sp_ids,current_sp_ids])
            
            
            
            if len(new_sp_ids) > 0:
                new_sp_TP = sub[np.isin(sub['sp_ID'],new_sp_ids)]['TP'].tolist()
                new_sp_Gen = sub[np.isin(sub['sp_ID'],new_sp_ids)]['Gen'].tolist()
                new_sp_Vul = sub[np.isin(sub['sp_ID'],new_sp_ids)]['Vul'].tolist()
                new_sp_TL = sub[np.isin(sub['sp_ID'],new_sp_ids)]['TL'].tolist()
                new_sp_BS = sub[np.isin(sub['sp_ID'],new_sp_ids)]['BS'].tolist()
            else:
                new_sp_ids = new_sp_TP = new_sp_Gen = new_sp_Vul = new_sp_TL = new_sp_BS = [np.nan]
            
            # print('new', new_sp_ids)
            
            cumulative_sum_value = len(np.unique(encountered_sp_ids))
            # Return the cumulative sum as a new column
            
            cumulative_biomass+= np.sum(sub['B_final'])
            
            result = pd.concat([result,
                                pd.DataFrame({'patch':patch,
                                              'restored_patch_seed': restored_patch_seed,
                                              'cumulative_new_sp_id': cumulative_sum_value,
                                              'cumulative_biomass':cumulative_biomass,
                                              'nb_improved': count_improved,
                                              'landscape_seed':landscape_seed,
                                              'sim':sim,
                                              'restoration_type': restoration_type,
                                              'dist_invasion': sub['dist_invasion'].unique()[0],
                                              'dist_improved': sub['dist_improved'].unique()[0],
                                              'deltaR': sub['deltaR'].unique()[0],
                                              
                                            
                                              'nb_new_sp':len(new_sp_ids),
                                              'new_sp_ids': new_sp_ids,
                                              'new_sp_BS': new_sp_BS,
                                              'new_sp_Gen': new_sp_Gen,
                                              'new_sp_Vul': new_sp_Vul,
                                              'new_sp_TL': new_sp_TL,
                                              'new_sp_TP': new_sp_TP
                                              }).reset_index()
                                ])

    return result

traits_new_colonisers = grouped.apply(get_traits_latest_recolonisers)
traits_new_colonisers = traits_new_colonisers.reset_index()


traits_new_colonisers_invadedPatch = traits_new_colonisers
palette = sb.color_palette("coolwarm", len(traits_new_colonisers_invadedPatch['new_sp_TL'].unique()))
# Create subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

# Plot for data1 with continuous lines on the first subplot
for i, tl in enumerate(np.sort(traits_new_colonisers_invadedPatch['new_sp_TL'].unique())):
    sb.pointplot(data=traits_new_colonisers_invadedPatch[traits_new_colonisers_invadedPatch['new_sp_TL'] == tl], 
                 x='nb_improved', y='new_sp_Gen', 
                 color=palette[i], order=[0,1,2,3,4,5], ax=axes[0])

# Customize the first plot
axes[0].set_ylabel("Mean generality of new recolonisers")
axes[0].set_xlabel("Number of improved patches")
axes[0].legend()
# axes[0].set_ylabel('Diet breadth of recoloniser')
# axes[0].set_xlabel('Mean food chain length of recolonised food web')

# Plot for data2 with dotted lines on the second subplot
for i, tl in enumerate(np.sort(traits_new_colonisers_invadedPatch['new_sp_TL'].unique())):
    axes[1].set_yscale('log')
    sb.pointplot(data=traits_new_colonisers_invadedPatch[traits_new_colonisers_invadedPatch['new_sp_TL'] == tl], 
                 x='nb_improved', y='new_sp_BS', 
                 color=palette[i], order=[0,1,2,3,4,5], ax=axes[1])

# Customize the second plot
axes[1].legend()
axes[1].set_ylabel("Mean body mass of new recolonisers (logged)")
axes[1].set_xlabel("Number of improved patches")

# Plot for data2 with dotted lines on the second subplot
for i, tl in enumerate(np.sort(traits_new_colonisers_invadedPatch['new_sp_TL'].unique())):
    sb.pointplot(data=traits_new_colonisers_invadedPatch[traits_new_colonisers_invadedPatch['new_sp_TL'] == tl], x='nb_improved', y='new_sp_Vul', 
                 color=palette[i], order=[0,1,2,3,4,5], ax=axes[2], label = tl)

# Customize the second plot
axes[2].legend(title = "Trophic level")
axes[2].set_ylabel("Mean vulnerability of new recolonisers")
axes[2].set_xlabel("Number of improved patches")

# Adjust layout
plt.tight_layout()
# plt.ylabel('Diet breadth of recoloniser')
# plt.xlabel('Mean food chain length of recolonised food web')
plt.savefig('D:/TheseSwansea/SFT/Figures/SuppFigure6.png', dpi = 400, bbox_inches = 'tight')


# %% Plots


# %%%% Restored vs non restored patches 

# looking at how biomass and species richness changed inside and away from restored patches


fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

sb.pointplot(res15_invasion_normal[(res15_invasion_normal['B_final'] > 0)], 
             x = 'deltaR',
             y = 'B_final',
             hue = 'nb_improved',
             palette = 'coolwarm',
             ax = ax1, linewidth=1)
res15_init_normal['deltaR'] = 0.5
res15_init_normal['nb_improved'] = 'Initial'
sb.pointplot(res15_init_normal[(res15_init_normal['B_final_init'] > 0)], 
             x = 'deltaR',
             y = 'B_final_init',
             hue = 'nb_improved',
             palette = 'coolwarm',
             ax = ax1, linewidth=1)
ax1.set_ylabel('Mean biomass')
ax1.set_xlabel('Patch quality')
ax1.legend(title = 'Number of improved patches', bbox_to_anchor = [2,1.2], ncol = 7)

FW15_init_normal['deltaR'] = 0.5
FW15_init_normal['nb_improved'] = 'Initial'
sb.pointplot(FW15_invasion_normal, 
             x = 'deltaR',
             y = 'S_local',
             hue = 'nb_improved',
             palette = 'coolwarm',
             ax = ax2, linewidth=1)
sb.pointplot(FW15_init_normal, 
             x = 'deltaR',
             y = 'S_local_init',
             hue = 'nb_improved',
             palette = 'coolwarm',
             ax = ax2, linewidth=1)
ax2.set_ylabel('Number of species')
ax2.set_xlabel('Patch quality')
ax2.legend().remove()

plt.savefig('D:/TheseSwansea/SFT/Figures/SuppFigure4.png', dpi = 400, bbox_inches = 'tight')

FW15_invasion_normal.groupby('deltaR').agg({'S_local':'mean'})

# %%%% Figure 2 - Biomass change with improvement

# Calculate total and mean biomass per patch grouped by trophic level
# for initial conditons
sum_init15 = res15_init_normal[res15_invasion_normal['B_final'] > 0].groupby([ # keep only species with positive biomass
    'sim','landscape_seed','patch','TL_init'], dropna = False).agg({'B_final_init':['sum', 'mean','count']}).reset_index()
sum_init15.columns = list(map(''.join, sum_init15.columns.values))
sum_init15['nb_improved'] = 'Initial'
FW15_init_normal['nb_improved'] = 'Initial'

# after recolonisation
sum_res15 = res15_invasion_normal[res15_invasion_normal['B_final'] > 0].groupby([
    'sim','restoration_type','nb_improved','landscape_seed','restored_patches_seed','patch','TL'], dropna = False).agg({'B_final':['sum', 'mean','count']}).reset_index()
sum_res15.columns = list(map(''.join, sum_res15.columns.values))


## different trophic levels
unique_categories = ['nb_top', 'nb_int', 'nb_herb', 'nb_plants']



##########################
## stats to write in paper
##########################

# biomass (patch-level)
sum_res15.groupby(['nb_improved','TL']).agg({'B_finalsum':'mean'}) ## change in mean biomass across trophic levels and improvement levels
sum_init15.groupby(['TL_init']).agg({'B_final_initsum':'mean'}) ## initial mean biomass across trophic levels 
sum_res15.groupby(['nb_improved','TL','restoration_type']).agg({'B_finalsum':'mean'}) ## change in mean biomass across trophic levels, improvement levels and restoration type (clustered x scattered)

# number of species (patch-level)
FW15_init_normal['nb_int_local_init'].describe() # number of intermediate species in initial simulations
FW15_init_normal['nb_top_local_init'].describe() # number of top species in initial simulations
FW15_invasion_normal.groupby(['nb_improved']).agg({ # number of species per trophic levels in restored simulations
    'nb_plants':'mean', 
    'nb_herb':'mean', 
    'nb_int':'mean', 
    'nb_top':'mean',
    'S_local':'mean',
    'persistence':'mean'}) 

# 13.884444 - 16.877037 # gain in int species
# 0.566667 - 1.000000


##########################
## setting up Figure 2
##########################

fig, ([ax2, ax3, ax7], [ax5, ax6, ax8]) = plt.subplots(nrows=2, ncols=3, figsize=(15,10), sharex=True)
fig.tight_layout(pad = 4)
colors = sb.color_palette("coolwarm", len(unique_categories))

## Change in TOTAL biomass across improvement levels from initial to 5 patches restored
### clustered landscape - full line
sb.pointplot(data = sum_res15[sum_res15['restoration_type'] != 'scattered'],  ## restored simulation 
                  y = 'B_finalsum', 
                  hue = 'TL', 
                  x = 'nb_improved', 
                  palette = sb.color_palette("coolwarm", len(unique_categories)),
                  estimator = 'mean',
                  ax = ax2, order = ['Initial',0,1,2,3,4,5])
sb.pointplot(data = sum_init15,  ## initial simulation 
                  y = 'B_final_initsum', 
                  hue = 'TL_init', 
                  x = 'nb_improved', 
                  palette = sb.color_palette("coolwarm", len(unique_categories)),
                  estimator = 'mean',
                  ax = ax2)
### scattered landscape = dotted (initial are the same as above, so no need to add them again)
sb.pointplot(data = sum_res15[sum_res15['restoration_type'] != 'clustered'], 
                  y = 'B_finalsum', 
                  hue = 'TL', 
                  x = 'nb_improved', 
                  palette = sb.color_palette("coolwarm", len(unique_categories)),
                  estimator = 'mean',
                  ax = ax2,
                  linestyles='dotted')
ax2.set_xlabel('Number of improved patches')
ax2.set_ylabel('Total biomass after recolonisation')
ax2.legend().remove()
ax2.annotate('A', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')

## Change in MEAN biomass across improvement levels from initial to 5 patches restored
## clustered landscape
sb.pointplot(data = sum_res15[sum_res15['restoration_type'] != 'scattered'], 
                  y = 'B_finalmean', 
                  hue = 'TL', 
                  x = 'nb_improved', 
                  palette = sb.color_palette("coolwarm", len(unique_categories)),
                  estimator = 'mean',
                  ax = ax5)
sb.pointplot(data = sum_init15, 
                  y = 'B_final_initmean', 
                  hue = 'TL_init', 
                  x = 'nb_improved', 
                  palette = sb.color_palette("coolwarm", len(unique_categories)),
                  estimator = 'mean',
                  ax = ax5)
## scattered landscape = dotted
sb.pointplot(data = sum_res15[sum_res15['restoration_type'] != 'clustered'], 
                  y = 'B_finalmean', 
                  hue = 'TL', 
                  x = 'nb_improved', 
                  palette = sb.color_palette("coolwarm", len(unique_categories)),
                  estimator = 'mean',
                  ax = ax5,
                  linestyles='dotted')
ax5.set_xlabel('Number of improved patches')
ax5.set_ylabel('Mean species biomass after recolonisation')
ax5.annotate('B', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')
ax5.legend().remove()


## Change in persistence across improvement levels from initial to 5 patches restored
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['restoration_type'] != 'scattered')], 
             y = 'persistence_int', 
             x = 'nb_improved',
             color = colors[2],
             ax = ax3)
sb.pointplot(data = FW15_init_normal, 
             y = 'persistence_int', 
             x = 'nb_improved',
             color = colors[2],
             ax = ax3)
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') &
                                         (FW15_invasion_normal['restoration_type'] != 'clustered')], 
             y = 'persistence_int', 
             x = 'nb_improved', 
             color = colors[2],
             ax = ax3,
             linestyles='dotted')
ax3.set_ylabel('Persistence of intermediate species'
               ''
               'after recolonisation')
ax3.set_xlabel('Number of improved patches')
ax3.annotate('C', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')

## Change in persistence across trophic levels 

## Persistence of top species - clustered
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') &
                                         (FW15_invasion_normal['restoration_type'] != 'scattered')], 
             y = 'persistence_top', 
             x = 'nb_improved',
             color = colors[3],
             ax = ax6)
sb.pointplot(data = FW15_init_normal, 
             y = 'persistence_top', 
             x = 'nb_improved',
             color = colors[3],
             ax = ax6)
## scattered
sb.pointplot(data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init') &
                                         (FW15_invasion_normal['restoration_type'] != 'clustered')], 
             y = 'persistence_top', 
             x = 'nb_improved',
             color = colors[3],
             ax = ax6,
             linestyles='dotted')
ax6.set_ylabel('Persistence of top species after recolonisation')
ax6.set_xlabel('Number of improved patches')
ax6.legend().remove()
ax6.annotate('D', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')



## barplots of extinctions and recolonisations

res15_invasion_normal['extinct'] = pd.Categorical(res15_invasion_normal['extinct']) # make 'extinct' a categorical variable
extinctions_dataframe = res15_invasion_normal.groupby(
    ['nb_improved','sim','restoration_type','landscape_seed','restored_patches_seed','patch','TL'], 
    dropna = False, observed = False
    )['extinct'].value_counts(normalize=False).reset_index(name='Count') # count the number of extinctions

# by setting extinct as a categorical this added un simulated rows from the clustered scenarios,
# combining clustered with non nan restored_patches_seed - we remove those to remove artificial zeros:
extinctions_dataframe = extinctions_dataframe[~((extinctions_dataframe['restoration_type'] == 'clustered') &
                                              (~np.isnan(extinctions_dataframe['restored_patches_seed'])))]


sb.barplot(data = extinctions_dataframe[extinctions_dataframe['extinct']],
           y = 'Count', x = 'nb_improved', hue = 'TL', 
           palette = sb.color_palette("coolwarm", len(unique_categories)), estimator = 'mean',
           ax = ax7)
ax7.set_ylabel('Number of extinctions after recolonisation')
ax7.set_xlabel('Number of improved patches')
ax7.annotate('E', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')
ax7.legend(bbox_to_anchor = [0.6, 1.3], title = 'Trophic level', ncols = 2)


## invasion dataframe
invasion_dataframe = res15_invasion_normal.copy()
# convert to categorical to conserve zero counts
invasion_dataframe['successful_invaders_initial_pop'] = pd.Categorical(invasion_dataframe['successful_invaders_initial_pop'])

invasion_dataframe = invasion_dataframe.groupby(
    ['nb_improved','sim','restoration_type','landscape_seed','restored_patches_seed','patch','TL'], dropna = False
    )['successful_invaders_initial_pop'].value_counts(normalize=False).reset_index(name='Count') # get number of invasions per patch per trophic level

# by setting successful_invaders_initial_pop as a categorical this added un simulated rows from the clustered scenarios
# by combining clustered with non nan restored_patches_seed - we remove those to remove artificial zeros
invasion_dataframe = invasion_dataframe[~((invasion_dataframe['restoration_type'] == 'clustered') &
                                          (~np.isnan(invasion_dataframe['restored_patches_seed'])))]


sb.barplot(data = invasion_dataframe[invasion_dataframe['successful_invaders_initial_pop']],
           y = 'Count', x = 'nb_improved', hue = 'TL', 
           palette = sb.color_palette("coolwarm", len(unique_categories)), estimator = 'mean',
           ax = ax8)
ax8.set_ylabel('Number of recolonising species')
ax8.set_xlabel('Number of improved patches')
ax8.annotate('F', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')
ax8.legend().remove()


plt.savefig('D:/TheseSwansea/SFT/Figures/Figure2.png', dpi = 400, bbox_inches = 'tight')



# %%%% Figure 3 - Random forests for number of recolonisers


invasions_byTL = res15_invasion_normal.copy()
## turm into categorical so that it includes zero counts
invasions_byTL['successful_invaders_initial_pop'] = pd.Categorical(invasions_byTL['successful_invaders_initial_pop'])
invasions_byTL = invasions_byTL.groupby(['sim','landscape_seed','restored_patches_seed',
                                         'nb_improved','patch','restoration_type','TL'], dropna = False, observed = False)['successful_invaders_initial_pop'].value_counts().reset_index(name='Count')

# remove clustered simulations that were added artifically hereand that all have zero counts
invasions_byTL = invasions_byTL[~((invasions_byTL['restoration_type'] == 'clustered') &
                                  (~np.isnan(invasions_byTL['restored_patches_seed'])))]


invasions_byTL = pd.merge(invasions_byTL, FW15_init_normal, on = ['landscape_seed','sim','patch'],
                          suffixes=['','_init'])
invasions_byTL = pd.merge(invasions_byTL, FW15_invasion_normal[['sim','nb_improved', 'landscape_seed', 'restored_patches_seed', 'patch', 'dist_invasion','dist_improved','latest_patch_improved']], 
                          on = ['sim','nb_improved', 'landscape_seed', 'restored_patches_seed','patch'], how = 'outer')

invasions_byTL = invasions_byTL[invasions_byTL['successful_invaders_initial_pop']]

invasions_byTL.loc[np.isnan(invasions_byTL['restored_patches_seed']),'restored_patches_seed'] = -1
invasions_byTL['restored_patches_seed'].unique()

invasions_byTL = pd.get_dummies(invasions_byTL, columns=['restoration_type']) ## create binary variable for restoration type (clustered/scattered)

invasions_byTL = pd.merge(invasions_byTL, landscape_characteristics[['distance_center', 'degree_patch', 'landscape_seed','Patch']], 
                          left_on = ['patch','landscape_seed'],
                          right_on = ['Patch','landscape_seed'])


invasions_byTL.to_csv(f'D:/TheseSwansea/Patch-Models/outputs/15Patches/InvasionDataFrameForR.csv')

import statsmodels.api as sm
from statsmodels.formula.api import glm

invasions_byTL['TL'] = pd.Categorical(invasions_byTL['TL'])
formula = 'Count ~ TL*dist_improved + TL*dist_invasion'
model = glm(formula=formula, data=invasions_byTL, family=sm.families.Poisson()).fit()
print(model.summary())

from pymer4.models import Lmer
formula = 'Count ~ TL*dist_improved + TL*dist_invasion + (1|landscape_seed)'
model = Lmer(formula, data=invasions_byTL, family='poisson')
result = model.fit()

# Print the model summary
print(result)

sb.scatterplot(invasions_byTL, x = 'C_local_init', y = 'S_local_init')
sb.scatterplot(invasions_byTL, x = 'LS_local_init', y = 'S_local_init')


### lookingh at number of invaders

## testing a random forest to predict invasion success of species
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TL_categories = invasions_byTL['TL'].unique()

colors = sb.color_palette("coolwarm", 10)
sb.palplot(colors)
colors = [(0.4570464785254902, 0.5940055499294118, 0.963029229690196),
          (0.8180564934117647, 0.8555896775450981, 0.9146376165490196),
          (0.9094595977529412, 0.8393864797647058, 0.8003313524235294),
          (0.9182816725843137, 0.48417347218039214, 0.37779392507058823)]
sb.palplot(colors)




plt.figure(figsize=(13, 8))

labels_panels = ['A','B','C','D']

# Loop through each TL category
for i, TL in enumerate(TL_categories):
    X = invasions_byTL[invasions_byTL['TL'] == TL][
        [# spatial considerations
         'dist_improved', 'dist_invasion', 'nb_improved',
         'restoration_type_scattered',
         'distance_center', 'degree_patch',
         
         ## local characteristics
         'LS_local_init', 'S_local_init', 'C_local_init',
         'StdGen_local_init', 'StdVul_local_init', 'MeanGen_local_init',
         'Mfcl_local_init', 
         'Modularity_local_init',
         'nb_top_local_init', 'nb_int_local_init', 
         'nb_herb_local_init', 'nb_plants_local_init'
        ]]
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        invasions_byTL[invasions_byTL['TL'] == TL]['Count'], test_size=0.2, random_state=42)
    
    # Initialize the Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 5)  # n_estimators is the number of trees
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf_model.predict(X_test)
    
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Trophic level {TL} Accuracy: {accuracy:.2f}")
    
    # # Detailed classification report
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    
    # # Confusion matrix
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for the importances
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    top_feature = feature_names[np.argmax(importances)]
    
    
    # Sort the DataFrame by importance
    feature_importances['Feature'] = feature_importances['Feature'].astype('category')
    feature_importances['Type'] = feature_importances['Feature'].map({'dist_improved':'Nutrient input',
                                                                                           'dist_invasion':'Patch location', 
                                                                                           'nb_improved': 'Nutrient input',
                                                                                           'landscape_seed':'Landscape',
                                                                                           'restored_patches_seed':'Landscape',
                                                                                           'distance_center':'Patch location',
                                                                                           'degree_patch':'Patch location',
                                                                                           'restoration_type_scattered':'Landscape',
                                                                                           
                                                                                           ## local characteristics
                                                                                           'Modularity_local_init': 'Local food web',
                                                                                           'S_local_init':'Local food web', 
                                                                                           'C_local_init':'Local food web',
                                                                                           'LS_local_init':'Local food web', 
                                                                                           'MeanGen_local_init':'Local food web', 
                                                                                           'StdVul_local_init':'Local food web', 
                                                                                           'StdGen_local_init':'Local food web', 
                                                                                           'Mfcl_local_init':'Local food web', 
                                                                                           'nb_top_local_init':'Local food web', 
                                                                                           'nb_int_local_init':'Local food web', 
                                                                                           'nb_herb_local_init':'Local food web', 
                                                                                           'nb_plants_local_init':'Local food web'})
    
    feature_importances['Feature_newNames'] = feature_importances['Feature'].map({'dist_improved':'Dist to closest improved patch',
                                                                                           'dist_invasion':'Dist to invaded patch', 
                                                                                           'nb_improved': 'Nb of improved patches',
                                                                                           'landscape_seed':'Landscape configuration',
                                                                                           'restored_patches_seed':'Restoration sequence',
                                                                                           'distance_center':'Dist to center of the landscape',
                                                                                           'degree_patch':'Euclidian distance to other patch',
                                                                                           'restoration_type_scattered':'Restoration clustering (scattered)',
                                                                                           
                                                                                           ## local characteristics
                                                                                           'Modularity_local_init': 'Modularity ',
                                                                                           'S_local_init':'Species richness ', 
                                                                                           'C_local_init':'Connectance ',
                                                                                           'LS_local_init':'Nb of links per species ', 
                                                                                           'MeanGen_local_init':'Mean generality ', 
                                                                                           'StdVul_local_init':'Sd of vulnerability ',
                                                                                           'StdGen_local_init':'Sd of generality ', 
                                                                                           'MeanTP_local_init':'Mean trophic position ', 
                                                                                           'Mfcl_local_init':'Mean food chain length ', 
                                                                                           'nb_top_local_init':'Nb of top species ', 
                                                                                           'nb_int_local_init':'Nb of intermediate species ', 
                                                                                           'nb_herb_local_init':'Nb of herbivore species ', 
                                                                                           'nb_plants_local_init':'Nb of plant species '})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    
    
    # Print the sorted feature importances
    print(feature_importances)
    
    
    # plt.figure(figsize = (12,12))
    # for j, feature in enumerate(feature_names):
        
    #     ax = plt.subplot(4, 5, j + 1)
    #     plt.tight_layout(pad = 1)
    
    #     sub = invasions_byTL[invasions_byTL['TL'] == TL]
    #     sub['Count'] = pd.Categorical(sub['Count']) # create categorical variable to get horizontal boxplot
    #     sb.boxplot(data=sub, y='Count', 
    #                 x=feature, ax=ax,
    #                 color='grey')    
    #     sb.pointplot(data=sub, y='Count', 
    #                 x=feature, ax=ax,
    #                 color='black', linestyles='None')    
    #     ax.set_xlabel(feature_importances['Feature_newNames'][j])
    #     ax.set_ylabel('Nb of recolonisers')
    #     ax.set_title(f'Importance score: {np.round(importances[j],2)}')
        
    # plt.savefig(f'D:/TheseSwansea/SFT/Figures/Effect-RF-TL{TL}.png', dpi = 400, bbox_inches = 'tight')

    
    
    # Plotting feature importances
    # Plotting feature importances
    ax = plt.subplot(2, 2, i + 1)
    plot = sb.barplot(x='Importance', y='Feature_newNames', data=feature_importances, 
                      hue='Type', palette=colors, order=feature_importances['Feature_newNames'],
                      hue_order=['Patch location', 'Landscape', 'Nutrient input','Local food web'], ax=ax)
    
    # Collect handles and labels once for the common legend
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
    
    ax.set_title(f'Trophic level {TL}')
    ax.set_xlabel('Importance')
    ax.set_ylabel('')
    ax.legend().remove()  # Remove individual legends
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.annotate(labels_panels[i], xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')

    
    # Create inset axes for boxplot
    axins = ax.inset_axes([0.5, 0.2, 0.4, 0.6])
    axins.patch.set_alpha(0.3)  # Set inset plot background to transparent
    
    sub = invasions_byTL[invasions_byTL['TL'] == TL]
    sub['Count'] = pd.Categorical(sub['Count']) # create categorical variable to get horizontal boxplot
    sb.boxplot(data=sub, y='Count', 
               x=top_feature, ax=axins,
               color='grey')    
    axins.tick_params(axis='x', labelsize=10)
    axins.tick_params(axis='y', labelsize=10)
    axins.set_xlabel(feature_importances['Feature_newNames'][np.argmax(importances)], fontsize=8)
    axins.set_ylabel('Nb of recolonisers', fontsize=8)

# Adjust layout
plt.tight_layout()
plt.figlegend(handles, labels, bbox_to_anchor = [0.82, 1.07], title='Drivers of recolonisation', ncol = 4)
plt.savefig('D:/TheseSwansea/SFT/Figures/RF-importance.png', dpi = 400, bbox_inches = 'tight')




# %%%% Random forest with just first patch improved (not in manuscript)


invasions_byTL_oneImprovement = invasions_byTL[invasions_byTL['nb_improved'] == 1]
invasions_byTL_oneImprovement = pd.merge(invasions_byTL_oneImprovement,landscape_characteristics[['distance_center', 'degree_patch', 'landscape_seed','Patch']], 
                          left_on = ['latest_patch_improved','landscape_seed'],
                          right_on = ['Patch','landscape_seed'], suffixes=['','_ImprovedPatch']
                                         )



plt.figure(figsize=(13, 8))

# Loop through each TL category
for i, TL in enumerate(TL_categories):
    X = invasions_byTL_oneImprovement[invasions_byTL_oneImprovement['TL'] == TL][
        [# spatial considerations
         'dist_improved', 'dist_invasion', 
         'restoration_type_scattered',
         'distance_center', 'degree_patch',
         'degree_patch_ImprovedPatch',
         
         ## local characteristics
         'LS_local_init', 'S_local_init',
         'StdGen_local_init', 'StdVul_local_init', 'MeanGen_local_init',
         'Mfcl_local_init', 
         'Modularity_local_init',
         'nb_top_local_init', 'nb_int_local_init', 
         'nb_herb_local_init', 'nb_plants_local_init'
        ]]
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        invasions_byTL_oneImprovement[invasions_byTL_oneImprovement['TL'] == TL]['Count'], test_size=0.2, random_state=42)
    
    # Initialize the Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 5)  # n_estimators is the number of trees
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf_model.predict(X_test)
    
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Trophic level {TL} Accuracy: {accuracy:.2f}")
    
    # # Detailed classification report
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    
    # # Confusion matrix
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for the importances
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    top_feature = feature_names[np.argmax(importances)]
    
    
    # Sort the DataFrame by importance
    feature_importances['Feature'] = feature_importances['Feature'].astype('category')
    feature_importances['Type'] = feature_importances['Feature'].map({'dist_improved':'Landscape',
                                                                                           'dist_invasion':'Landscape', 
                                                                                           'nb_improved': 'Landscape',
                                                                                           'landscape_seed':'Landscape',
                                                                                           'restored_patches_seed':'Landscape',
                                                                                           'distance_center':'Landscape',
                                                                                           'degree_patch':'Landscape',
                                                                                           'degree_patch_ImprovedPatch':'Landscape',
                                                                                           'restoration_type_scattered':'Landscape',
                                                                                           
                                                                                           ## local characteristics
                                                                                           'Modularity_local_init': 'Local food web',
                                                                                           'S_local_init':'Local food web', 
                                                                                           'C_local_init':'Local food web',
                                                                                           'LS_local_init':'Local food web', 
                                                                                           'MeanGen_local_init':'Local food web', 
                                                                                           'StdVul_local_init':'Local food web', 
                                                                                           'StdGen_local_init':'Local food web', 
                                                                                           'Mfcl_local_init':'Local food web', 
                                                                                           'nb_top_local_init':'Local food web', 
                                                                                           'nb_int_local_init':'Local food web', 
                                                                                           'nb_herb_local_init':'Local food web', 
                                                                                           'nb_plants_local_init':'Local food web'})
    
    feature_importances['Feature_newNames'] = feature_importances['Feature'].map({'dist_improved':'Dist to closest improved patch',
                                                                                           'dist_invasion':'Dist to invaded patch', 
                                                                                           'nb_improved': 'Nb of improved patches',
                                                                                           'landscape_seed':'Landscape configuration',
                                                                                           'restored_patches_seed':'Restoration sequence',
                                                                                           'distance_center':'Dist to center of the landscape',
                                                                                           'degree_patch':'Euclidian distance to other patch',
                                                                                           'degree_patch_ImprovedPatch':'Euclidian distance of improved patch to other patches',
                                                                                           'restoration_type_scattered':'Restoration clustering (scattered)',
                                                                                           
                                                                                           ## local characteristics
                                                                                           'Modularity_local_init': 'Modularity ',
                                                                                           'S_local_init':'Species richness ', 
                                                                                           'C_local_init':'Connectance ',
                                                                                           'LS_local_init':'Nb of links per species ', 
                                                                                           'MeanGen_local_init':'Mean generality ', 
                                                                                           'StdVul_local_init':'Sd of vulnerability ',
                                                                                           'StdGen_local_init':'Sd of generality ', 
                                                                                           'MeanTP_local_init':'Mean trophic position ', 
                                                                                           'Mfcl_local_init':'Mean food chain length ', 
                                                                                           'nb_top_local_init':'Nb of top species ', 
                                                                                           'nb_int_local_init':'Nb of intermediate species ', 
                                                                                           'nb_herb_local_init':'Nb of herbivore species ', 
                                                                                           'nb_plants_local_init':'Nb of plant species '})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    
    
    # Print the sorted feature importances
    print(feature_importances)
    
    
    # plt.figure(figsize = (12,12))
    # for j, feature in enumerate(feature_names):
        
    #     ax = plt.subplot(4, 5, j + 1)
    #     plt.tight_layout(pad = 1)
    
    #     sub = invasions_byTL_oneImprovement[invasions_byTL_oneImprovement['TL'] == TL]
    #     sub['Count'] = pd.Categorical(sub['Count']) # create categorical variable to get horizontal boxplot
    #     sb.boxplot(data=sub, y='Count', 
    #                 x=feature, ax=ax,
    #                 color='grey')    
    #     sb.pointplot(data=sub, y='Count', 
    #                 x=feature, ax=ax,
    #                 color='black', linestyles='None')    
    #     ax.set_xlabel(feature_importances['Feature_newNames'][j])
    #     ax.set_ylabel('Nb of recolonisers')
    #     ax.set_title(f'Importance score: {np.round(importances[j],2)}')
        
    # plt.savefig(f'D:/TheseSwansea/SFT/Figures/Effect-1Improvement-RF-TL{TL}.png', dpi = 400, bbox_inches = 'tight')

    
    
    # Plotting feature importances
    # Plotting feature importances
    ax = plt.subplot(2, 2, i + 1)
    plot = sb.barplot(x='Importance', y='Feature_newNames', data=feature_importances, 
                      hue='Type', palette=colors, order=feature_importances['Feature_newNames'],
                      hue_order=['Landscape', 'Local food web'], ax=ax)
    
    # Collect handles and labels once for the common legend
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
    
    ax.set_title(f'Trophic level {TL}')
    ax.set_xlabel('Importance')
    ax.set_ylabel('')
    ax.legend().remove()  # Remove individual legends
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Create inset axes for boxplot
    axins = ax.inset_axes([0.5, 0.2, 0.4, 0.6])
    axins.patch.set_alpha(0.3)  # Set inset plot background to transparent
    
    sub = invasions_byTL_oneImprovement[invasions_byTL_oneImprovement['TL'] == TL]
    sub['Count'] = pd.Categorical(sub['Count']) # create categorical variable to get horizontal boxplot
    sb.boxplot(data=sub, y='Count', 
               x=top_feature, ax=axins,
               color='grey')    
    axins.tick_params(axis='x', labelsize=10)
    axins.tick_params(axis='y', labelsize=10)
    axins.set_xlabel(feature_importances['Feature_newNames'][np.argmax(importances)], fontsize=8)
    axins.set_ylabel('Nb of recolonisers', fontsize=8)

# Adjust layout
plt.tight_layout()
plt.figlegend(handles, labels, bbox_to_anchor = [0.68, 1.05], title='Feature Type', ncol = 3)
# plt.savefig('D:/TheseSwansea/SFT/Figures/RF-importance.png', dpi = 400, bbox_inches = 'tight')



# %%%% distance from invaded patch (not in manuscript)

fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
fig.tight_layout(pad=3.0)

data = FW15_invasion_normal[(FW15_invasion_normal['stage'] != 'init')]
unique_categories = data['nb_improved'].unique()
colors = sb.color_palette("coolwarm", len(unique_categories))

for i, category in enumerate(unique_categories):
    category_data = data[data['nb_improved'] == category]
    sb.regplot(data=category_data, y='nb_invaders_initial_pop', x='dist_invasion', 
               order=2, color=colors[i], label=f'{category}',
               ax = ax1)
ax1.legend().remove()
ax1.set_ylabel('Number of recolonising species')
ax1.set_xlabel('Distance from invaded patch')
ax1.annotate('F', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')
ax1.legend(bbox_to_anchor = [0.8, 1.3], title = 'Number of patches improved', ncols = 3)


for i, category in enumerate(unique_categories):
    category_data = data[data['nb_improved'] == category]
    sb.regplot(data=category_data, y='nb_invaders_initial_pop', x='dist_improved', 
               order=2, color=colors[i], label=f'{category}')
ax2.legend().remove()
ax2.set_ylabel('Number of recolonising species')
ax2.set_xlabel('Distance from closest improved patch')
ax2.annotate('F', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')


plt.savefig('D:/TheseSwansea/SFT/Figures/SupplementaryFig3.png', dpi = 400, bbox_inches = 'tight')




# %%%% Figure 4 - Change in food web metrics - local properties

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(nrows=2, ncols=3, figsize=(12,9), sharex=True)
fig.tight_layout(pad=3.0)

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'scattered'], 
              y = 'S_local', x = 'nb_improved', ax = ax1, color = 'black')
sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'clustered'], 
              y = 'S_local', x = 'nb_improved', ax = ax1, color = 'grey', linestyles='dotted')
ax1.set_ylabel('Local species richness')
ax1.set_xlabel('Number of improved patches')

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'scattered'], 
              y = 'L_local', x = 'nb_improved', ax = ax2, color = 'black')
sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'clustered'], 
              y = 'L_local', x = 'nb_improved', ax = ax2, color = 'grey', linestyles='dotted')
ax2.set_ylabel('Number of links')
ax2.set_xlabel('Number of improved patches')

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'scattered'], 
              y = 'MeanBodyMass_local', x = 'nb_improved', ax = ax3, color = 'black')
sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'clustered'], 
              y = 'MeanBodyMass_local', x = 'nb_improved', ax = ax3, color = 'grey', linestyles='dotted')
ax3.set_ylabel('Mean body mass')
ax3.set_xlabel('Number of improved patches')

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'scattered'], 
              y = 'Mfcl_local', x = 'nb_improved', ax = ax4, color = 'black')
sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'clustered'], 
              y = 'Mfcl_local', x = 'nb_improved', ax = ax4, color = 'grey', linestyles='dotted')
ax4.set_ylabel('Mean food chain length')
ax4.set_xlabel('Number of improved patches')

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'scattered'], 
              y = 'LS_local', x = 'nb_improved', ax = ax5, color = 'black')
sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'clustered'], 
              y = 'LS_local', x = 'nb_improved', ax = ax5, color = 'grey', linestyles='dotted')
ax5.set_ylabel('Number of links per species')
ax5.set_xlabel('Number of improved patches')

sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'scattered'], 
              y = 'alpha_diversity_shannon', x = 'nb_improved', ax = ax6, color = 'black')
sb.pointplot(data = FW15_invasion_normal[FW15_invasion_normal['restoration_type'] != 'clustered'], 
              y = 'alpha_diversity_shannon', x = 'nb_improved', ax = ax6, color = 'grey', linestyles='dotted')
ax6.set_ylabel('Alpha diversity (Shannon diversity)')
ax6.set_xlabel('Number of improved patches')


plt.savefig('D:/TheseSwansea/SFT/Figures/Local-FW-Chara.png', dpi = 400, bbox_inches = 'tight')


# %%%% Change in food web metrics - regional properties (not in manuscript)

fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(nrows=3, ncols=2, figsize=(12,9), sharex=True)
fig.tight_layout(pad=3.0)

sb.lineplot(data = FW15_invasion_normal, 
              y = 'S_regional', x = 'nb_improved', ax = ax1, color = 'black')
ax1.set_ylabel('Local species richness')
ax1.set_xlabel('Number of improved patches')

sb.lineplot(data =  FW15_invasion_normal, 
              y = 'L', x = 'nb_improved', ax = ax2, color = 'black')
ax2.set_ylabel('Numner of links')
ax2.set_xlabel('Number of improved patches')

sb.lineplot(data =  FW15_invasion_normal, 
              y = 'C', x = 'nb_improved', ax = ax3, color = 'black')
ax3.set_ylabel('Connectance')
ax3.set_xlabel('Number of improved patches')

sb.lineplot(data =  FW15_invasion_normal, 
              y = 'Modularity', x = 'nb_improved', ax = ax4, color = 'black')
ax4.set_ylabel('Modularity')
ax4.set_xlabel('Number of improved patches')

sb.lineplot(data =  FW15_invasion_normal, 
            y = 'beta_diversity_shannon', x = 'nb_improved', ax = ax5, color = 'black')
ax5.set_ylabel('Beta diversity')
ax5.set_xlabel('Number of improved patches')

sb.lineplot(data =  FW15_invasion_normal, 
              y = 'gamma_diversity_shannon', x = 'nb_improved', ax = ax6, color = 'black')
ax6.set_ylabel('Gamma diversity')
ax6.set_xlabel('Number of improved patches')


plt.savefig('D:/TheseSwansea/SFT/Figures/Regional-FW-Chara.png', dpi = 400, bbox_inches = 'tight')


# %%%% Traits of invaders (not in manuscript)


invasion_dataframe = res15_invasion_normal[(~res15_invasion_normal['successful_invaders_initial_pop'].isna()) &
                                           (res15_invasion_normal['dist_invasion'] == 0)].groupby(['TL','patch', 'nb_improved','sim','restoration_type','landscape_seed','restored_patches_seed'], dropna = False)['successful_invaders_initial_pop'].value_counts(normalize=True).mul(100).reset_index(name='Percentage')

test = res15_invasion_normal[(~res15_invasion_normal['successful_invaders_initial_pop'].isna())].groupby(
    ['nb_improved','sim','restoration_type','landscape_seed','restored_patches_seed','TL'], dropna = False).agg({'B_final': lambda x: len(np.where(x>0)[0]), 'B_init_init': lambda x: len(np.where(x>0)[0]),})



invasion_dataframe = pd.merge(invasion_dataframe, landscape_characteristics, left_on = ['landscape_seed','patch'], right_on = ['landscape_seed','Patch'])

sb.lmplot(invasion_dataframe, x = 'degree_patch', y = 'Percentage', hue = 'sim')


FW15_invasion_normal = pd.merge(FW15_invasion_normal, landscape_characteristics, left_on = ['landscape_seed','patch'], right_on = ['landscape_seed','Patch'])

sb.lmplot(FW15_invasion_normal, x = 'degree_patch', y = 'nb_invaders_initial_pop', hue = 'nb_improved')

test = res15_invasion_normal[(~res15_invasion_normal['successful_invaders_initial_pop'].isna()) &
                                           (res15_invasion_normal['dist_invasion'] == 0) &
                                           (res15_invasion_normal['Invaders'])].groupby(['patch', 'nb_improved','sim','restoration_type','landscape_seed','restored_patches_seed', 'successful_invaders_initial_pop'], dropna = False).agg({'TL':'mean','Gen':['mean','size']})


### looking at invasion success and the traits if inavders

preinvasion = res15_init_normal[(res15_init_normal['B_final_init'] > 0)]
preinvasion = pd.merge(preinvasion, FW15_init_normal, on = ['sim','landscape_seed','patch'])


invaded_patch_invaders_5PatchesRestored = res15_invasion_normal[(~res15_invasion_normal['successful_invaders_initial_pop'].isna()) &
                                           (res15_invasion_normal['dist_invasion'] == 0) &
                                           (res15_invasion_normal['Invaders']) &
                                           (res15_invasion_normal['nb_improved'] == 5)]

invaded_patch_invaders = res15_invasion_normal[(~res15_invasion_normal['successful_invaders_initial_pop'].isna()) &
                                           (res15_invasion_normal['dist_invasion'] == 0) &
                                           (res15_invasion_normal['Invaders'])]
## /!\ maybe instead of invaders do all species that had zero biomass before invasion?

# %%%%% plotting invader traits depending on whether they were successful or not

unique_categories = ['nb_top', 'nb_int', 'nb_herb', 'nb_plants']


fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, figsize=(7,6))
fig.tight_layout(pad = 3)

unique_categories = invaded_patch_invaders['TL'].unique()
colors = sb.color_palette("coolwarm", len(unique_categories))

for i, category in enumerate(unique_categories):
    category_data = invaded_patch_invaders[(invaded_patch_invaders['TL'] == category) &
                                           (invaded_patch_invaders['successful_invaders_initial_pop'])]
    sb.pointplot(category_data, x = 'nb_improved', y = 'Gen',
                 color=colors[i],
                 label = f'{category}')
    
    category_data = invaded_patch_invaders[(invaded_patch_invaders['TL'] == category) &
                                           (~invaded_patch_invaders['successful_invaders_initial_pop'])]
    sb.pointplot(category_data, x = 'nb_improved', y = 'Gen',
                 color=colors[i],
                 linestyles='dotted')
    
for i, category in enumerate(unique_categories):
    plt.yscale('log')
    category_data = invaded_patch_invaders[(invaded_patch_invaders['TL'] == category) &
                                           (invaded_patch_invaders['successful_invaders_initial_pop'])]
    sb.pointplot(category_data, x = 'nb_improved', y = 'BS',
                 color=colors[i],
                 label = f'{category}')
    
    category_data = invaded_patch_invaders[(invaded_patch_invaders['TL'] == category) &
                                           (~invaded_patch_invaders['successful_invaders_initial_pop'])]
    sb.pointplot(category_data, x = 'nb_improved', y = 'BS',
                 color=colors[i],
                 linestyles='dotted')
    
    
    
ax3.legend(bbox_to_anchor = [0.8, 1.3], title = 'Number of patches improved', ncols = 3)
ax3.set_ylabel('Number of recolonising species')
ax3.set_xlabel('Modularity of invaded food web')
ax3.annotate('E', xy = (-0.1,1.1), verticalalignment='top', xycoords='axes fraction', fontsize='large')
## looking at how traits of invaders change with food web topolgy


fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, figsize=(7,6))
fig.tight_layout(pad = 3)

sb.pointplot(preinvasion, y = 'Gen', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax1, marker="+")
sb.pointplot(invaded_patch_invaders, x = 'successful_invaders_initial_pop', y = 'Gen', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax1)
ax1.set_ylabel('Generality')
ax1.set_xlabel('Invasion success')
ax1.legend(bbox_to_anchor = [1.7,1.2], ncol = 4)

ax2.set_yscale('log')
sb.pointplot(invaded_patch_invaders, x = 'successful_invaders_initial_pop', y = 'BS', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax2)
sb.pointplot(preinvasion, y = 'BS', hue = 'TL',
              palette = sb.color_palette("coolwarm", len(unique_categories)),
              ax = ax2, marker="+")
ax2.set_ylabel('Body mass')
ax2.set_xlabel('Invasion success')
ax2.legend().remove()

sb.pointplot(invaded_patch_invaders, x = 'successful_invaders_initial_pop', y = 'TP', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax3)
sb.pointplot(preinvasion, y = 'TP', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax3, marker="+")
ax3.set_ylabel('Trophic position')
ax3.set_xlabel('Invasion success')
ax3.legend().remove()

sb.pointplot(invaded_patch_invaders, x = 'successful_invaders_initial_pop', y = 'Vul', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax4)
sb.pointplot(preinvasion, y = 'Vul', hue = 'TL',
             palette = sb.color_palette("coolwarm", len(unique_categories)),
             ax = ax4, marker="+")
ax4.set_ylabel('Vulnerability')
ax4.set_xlabel('Invasion success')
ax4.legend().remove()

plt.savefig('D:/TheseSwansea/SFT/Figures/Figure4.png', dpi = 400, bbox_inches = 'tight')



