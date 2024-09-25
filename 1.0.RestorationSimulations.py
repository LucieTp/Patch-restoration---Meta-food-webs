# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:08:31 2023

@author: lucie.thompson
"""


###############################################################################
### SIMPLE 3 patches simulation - O - O - O
## 1 - No protected area O - O - O
## 2 - One protected area in the middle O - (O) - O

import pandas as pd 
## pandas needs to be version 1.5.1 to read the npy pickled files with np.load() 
## created under this version 
## this can be checked using pd.__version__ and version 1.5.1 can be installed using
## pip install pandas==1.5.1 (how I did it on SLURM - 26/03/2024)
## if version 2.0 or over yields a module error ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'
## I think to solve this, would need to run the code and save files under version > 2.0 of pandas
## or manually create this pandas.core.index.numerical function see issue in https://github.com/pandas-dev/pandas/issues/53300

from scipy.integrate import odeint, solve_ivp # Numerical integration function from the scientific programming package

import matplotlib.pyplot as plt 

import numpy as np # Numerical objects

import time

import networkx as nx

import igraph as ig

from argparse import ArgumentParser # for parallel runnning on Slurm

import pickle

import seaborn as sb

### this function implements the niche model by Williams and Martinez (2000) 
### given two params: S and C it yields food webs based on the distance between 
### species' niches
def nicheNetwork(S,C):
    
    x = 0
    y = S
    dup_sp = False
    
    while x < S or y > 1 or dup_sp :
         dup_sp = False
         M = np.zeros(shape = (S,S))
         
         # first we obtain the n values for all the species 
         # (ordered so we know the first sp is a basal species)
         n = np.sort(np.random.uniform(0,1,S))
         
         #we then obtain the feeding range for each of the species, drawn from a beta
         #distribution with beta value = (1/(2*C)) - 1
         beta = (1/(2*C)) - 1
         r = np.random.beta(1, beta, S) * n
         
         #we enforce the species with the lowest niche value to be a basal species
         r[0] = 0
         
         #the centre of the feeding range is drawn from a uniform distribution
         #between ri/2 and min(ni, 1-ri/2)
         c = np.zeros(S)
         for i in range(S):
             c[i] = np.random.uniform(r[i]/2,min(n[i], 1-r[i]/2))
             
             offset = r[i]/2
             upper_bound = c[i] + offset
             lower_bound = c[i] - offset
             
             for j in range(S):
                 if n[j] > lower_bound and n[j] < upper_bound:
                     M[j,i] = 1
                     
         #we verify that the network (i) is connected and (2) does not have cycles with no external energy input
         M_temp = M.copy()
         np.fill_diagonal(M_temp,0)
    
         graf = ig.Graph.Adjacency(M)
         y = graf.components(mode='weak')
         y = y._len
         if y > 1: # number of clusters
             next
    
         clts = graf.components(mode='strong')
         x = clts._len # number of clusters
         if x < S:
             clts_nos = [i for i in range(len(clts.sizes())) if clts.size(i) > 1]
             cycles_no_input = False
             
             for cn in clts_nos:
                 members = [i for i in range(len(clts.membership)) if clts.membership[i] == cn]
                 cluster_ok = False
            
                 for m in members:
                    prey = graf.neighbors(m, mode='in')
                    
                    if len(np.intersect1d(prey,members))  > len(members) + 1:
                        ## if this happens, this cycle/cluster has external energy input
                        cluster_ok = True
                        break
                
                    if not cluster_ok:
                       #print("NicheNetwork :: the network has cycles with no external energy input...");
                       cycles_no_input = True
                       break
       
      
             if cycles_no_input:
                 next
             else:
                 x = S
      
    
         #and we also verify that there are not duplicate species
         for i in range(S):
             if dup_sp:
                 break
          
             preys_i = M[:,i]
             predators_i = M[i,:]
             for j in range(S):
                if i == j:  
                    next
                sim_prey = preys_i == M[:,j]
                sim_preds = predators_i == M[j,:]
                
                if sum(sim_prey) == S and sum(sim_preds) == S:
                  dup_sp <- True
                  #print("NicheNetwork :: there is a duplicate species");
                  break
    
          
              #as long as we are traversing the network we might as well check
              #for basal species with cannibalistic links...
          
             if sum(preys_i) == 1 and M[i,i] == 1:
                 #print("NicheNetwork :: a cannibalistic link was found in a basal species... removing it");
                 M[i,i] = 0
        
    #we keep a reference of the original niche as a way of idetifying the species in the network:
    #each species is going to be identified by the index of its niche value within the 
    #original_niche array.
    original_niche = n
  
  
    ########################################################################################
  ## c: centre of the feeding range, r: distance from centre of the feeding range
  ## n: niche of the species

    return {'Stot':len(M), 'C':M.sum()/(len(M)**2), 'M':M, 'Niche':n, 'Centre':c, 'radius':r, 'Original_C':C, 'original_niche':original_niche}



###############################################################################
## ADDING BODY SIZE BASED ON TROPHIC POSITION 

## heavily inspired by Sentis et al 2020 (warming and invasion in food webs)
## r code at https://github.com/mlurgi/temperature-dependent-invasions

def NormaliseMatrix(M):
    colsum = sum(M)
    colsum[colsum == 0] = 1
    return M/colsum


def TrophicLevels(FW, TP):
    S = FW['Stot']
    
    if S<2:
        return FW
    
    TL = np.repeat(-1, S)
    
    M_temp = FW['M']
    np.fill_diagonal(M_temp,0) 
    
    ## species w/ TP 1 are basal species
    TL[np.array(TP) == 1] = 0
    
    for i in range(S):
        if TL[i] == -1:
            herb = True
            top = True
            
            # 1 - if the species only has prey in TL 0 then it is a herbivore
            if sum(TL[M_temp[:,i] != 0]) != 0:
                herb = False
            
            # 2 - if it has any predators then not a top pred
            if sum(M_temp[i,:]) > 0:
                top = False
            
            if herb:
                TL[i] = 1
            elif top:
                TL[i] = 3
            else:
                TL[i] = 2
    
    return TL


##### prey-averaged trophic level metric:
##### Williams and Martinez (2004). American Naturalist.
##### Limits to trophic levels and omnivory in complex food webs: theory and data.

def trophicPositions(FW):
    S = FW['Stot']
    
    if S<3: 
        return FW
    
    M = NormaliseMatrix(FW['M'])
    
    ## solves (I - A) * X = B
    # this means that species with no prey get TP = 1
    if np.linalg.det(np.diag(np.repeat(1,S)) - np.transpose(M)) != 0: 
        TP = np.linalg.solve(np.diag(np.repeat(1,S)) - np.transpose(M), np.repeat(1,S))
    
    else:
        tmp = np.diag(np.repeat(1,S))
        for i in range(9):
            tmp = np.matmul(tmp,np.transpose(M)) + np.diag(np.repeat(1,S))
        TP = np.matmul(tmp, np.repeat(1,S))
    W = M
    TL = TrophicLevels(FW, TP)
    
    FW.update({'TL':TL,'TP':TP,'NormM':W})
    return FW

    
def obtainBodySize(FW):
    
    FW = trophicPositions(FW)
    # average predator prey mass ratio
    ppmr = 100
    
    BS = 0.01*(ppmr**(FW['TP'] - 1))
    
    FW.update({'BS':BS})
    return FW

    

def create_landscape(P, extent, radius_max, nb_center, seed):
    
    '''
    Create landscape (x and y coordinates of each patch with outside and middle patches.
                      
    nb_center middle patches are randomly scattered within a circle of radius_max 
    centered around the central coordinates of the landscape.
    Outside patches must fall outside of this circle. 
    
    Input the number of patch and extent of the landscape (consider that all lanscapes are 
                                                           square of dimensions LxL, thus extent 
                                                           corresponds to length of one edge of the square (= L))
    

    Randomly draws x and y coordinates from a uniform distribution bounded by the extent given. 
    We seed the random number generation to have the same landscape configuration at each run. 
        
    Returns pandas dataframe with x and y columns and patch ID
    
    '''
    
    coords = pd.DataFrame()        
    
    for i in range(P):
        seed_index = seed[i]
        np.random.seed(seed_index)
        
        # if we have already drawn all the center point, we just draw random coordinates outside of the center
        ## outside patches
        if (i >= nb_center):
            
            y = np.random.uniform(extent[0], extent[1])
            x = np.random.uniform(extent[0], extent[1])
            
            # convert to polar
            r = np.sqrt((x - np.mean(extent))**2 + (y - np.mean(extent))**2)
            
            while(r < radius_max): # while the coordinates fall within the radius of the center points we resample coordinates
                
                seed_index+=1 # change random seed to get new set of coordinates
                np.random.seed(seed_index)

                x = np.random.uniform(extent[0], extent[1])
                y = np.random.uniform(extent[0], extent[1])
                
                ## convert to polar and check radius again
                r = np.sqrt((x - np.mean(extent))**2 + (y - np.mean(extent))**2)

            coords = pd.concat((coords, pd.DataFrame({'Patch':[i], 'x':x, 'y':y, 'position': 'outside', 'seed':seed[i]})))
        
        ## middle patches
        else:
            
            ## draw a random radius and a random angle 
            # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
            r = radius_max * np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            
            # convert from polar to cartesian (referential = center of the landscape)
            x = np.mean(extent) + r * np.cos(theta)
            y = np.mean(extent) + r * np.sin(theta)

            coords = pd.concat((coords, pd.DataFrame({'Patch':[i],'x':x, 'y':y, 'position': 'center', 'seed':seed[i]})))
    
    ## check that no distances are too small
    dist = get_distance(coords)
    while dist[(dist > 0) & (dist < 0.01)].size > 0:
        patch_to_redraw = np.where((dist > 0) & (dist < 0.01))[0][0]
        if coords[coords['Patch'] == patch_to_redraw]['position'][0] == 'outside':
            
            y = np.random.uniform(extent[0], extent[1])
            x = np.random.uniform(extent[0], extent[1])
            
            # convert to polar
            r = np.sqrt((x - np.mean(extent))**2 + (y - np.mean(extent))**2)
            
            while(r < radius_max): # while the coordinates fall within the radius of the center points we resample coordinates
                
                seed_index+=1 # change random seed to get new set of coordinates
                np.random.seed(seed_index)

                x = np.random.uniform(extent[0], extent[1])
                y = np.random.uniform(extent[0], extent[1])
                
                ## convert to polar and check radius again
                r = np.sqrt((x - np.mean(extent))**2 + (y - np.mean(extent))**2)

            coords.iloc[patch_to_redraw,:] =  [patch_to_redraw, x, y, 'outside', seed_index]
        
        else:
            
            seed_index+=1 # change random seed to get new set of coordinates
            np.random.seed(seed_index)
            
            ## draw a random radius and a random angle 
            # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
            r = radius_max * np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            
            # convert from polar to cartesian (referential = center of the landscape)
            x = np.mean(extent) + r * np.cos(theta)
            y = np.mean(extent) + r * np.sin(theta)

            coords.iloc[patch_to_redraw,:] = [patch_to_redraw, x, y, 'center', seed_index]

        dist = get_distance(coords)
    
    
    ### plot landscape map  
    palette_colors = sb.color_palette("viridis", 15)
    coords['Patch'] = coords['Patch'].astype('category')
    sb.scatterplot(data = coords, x = 'x',y = 'y', hue = 'Patch', style = 'position', palette = palette_colors)
    plt.scatter(x = np.mean(extent), y = np.mean(extent), s = 100, marker = 'X', c = 'black') # center of the landscape
    plt.show()
    
    return(coords)



def get_distance(coords):
    
    '''
    Calculate euclidian distance between patches whose coordinates are 
    recorded in coords (should have an 'x' and 'y' column with coordinates).
    
    Returns a PxP matrix for all possible pairwise distances.
    
    '''
    
    P = coords.shape[0]
    dist = np.zeros((P,P))
    for p1 in range(P): 
        for p2 in range(P):
            dist[p1,p2] = ((coords['x'].iloc[p2] - coords['x'].iloc[p1])**2 + (coords['y'].iloc[p2] - coords['y'].iloc[p1])**2)**(1/2)
    
    return(dist)

          
params = {## attack rate parameters - alphas (Binzer et al)
    "ab":0.25, # scale to body mass of resource species
    "ad":np.exp(-13.1),
    "ac":-0.8, # scale to body mass of consumer species
    
    ## handling time parameters - th (Binzer et al)
    "hd":np.exp(9.66),
    "hb":-0.45, # scale to body mass of ressource species
    "hc":0.47, # scale to body mass of consumer species
    
    # growth rate params
    "rd":np.exp(-15.68), "rb":-0.25,
    
    # metabolic rate parameters
    "xd":np.exp(-16.54), "xb":-0.31,
    
    # carrying capacity params
    "Kb":0.28, "Kd":5, 
    
    ## maximum dispersal distance (Häussler et al 2021)
    "d0": 0.1256, "eps":0.05} ## might need to change those paramaters if we want them to range exactly from 0.158 to 0.5 as for Johanna's study

      
def getSimParams(FW,S,P,coords,extent,params=params):
    
    FW = obtainBodySize(FW)
    
    Stot = FW['Stot']
    M = FW['M']
    BS = FW['BS']
    
    sp_ID = np.arange(Stot)
        
    # intialise matrixes 
    a = np.zeros((Stot,Stot)) # attack rates
    h = np.zeros((Stot,Stot)) # handling time
    
    # BSr = np.zeros((Stot,Stot)) # just for plotting
             
    # # allometric constants
    # ar = 1
    # ax = 0.88 # metabolic rate for ectotherms
    BSp = 0.01 # body mass of producer
    
    ## body mass dependant metabolic rate of predators, 0.138 (Ryser et al. 2021) for basal species aka plants
    # x = np.array([0.138 if sum(a[:,i]) == 0 else ax*(BS[i]**(-0.25))/(ar*(BSp**(-0.25))) for i in range(len(a))])
    
    ## body mass dependant metabolic rate for consumers only
    x = np.array([params['xd']*BS[i]**params['xb'] if sum(M[:,i]) > 0 else 0 for i in range(Stot)])
    # plt.scatter(x,[math.log(i) for i in BS])
    # plt.show() 
    
    ## growth rate (only for plants)
    r = [params['rd']*BSp**params['rb'] if sum(M[:,i]) == 0 else 0 for i in range(Stot)]
    # equal patch quality
    r = np.tile(r,P).reshape(P,Stot)
    
    K = np.array([params["Kd"]*BSp**params["Kb"] if sum(M[:,i]) == 0 else 0 for i in range(Stot)])
    
    # initial conditions - first we randomly allocate the species present in each patch
    y0 = [[1]*Sp + [0]*(Stot-Sp) for Sp in S]
    for init in range(len(y0)):
        np.random.seed(init)
        np.random.shuffle(y0[init])
        
    # then we adjust the initial biomass (=K for producers, K/8 for consumers)
    y0 = y0*np.array([k if k!=0 else K[K!=0].mean()/8 for k in K])
    
    # calculate attack rates and handling times that scale with body size
    # j: predator, i: prey
    # for i in range(Stot):
    #     for j in range(Stot):
    #         if M[i,j]>0: # j eats i
    #             # BSr[i,j] = BS[j]/BS[i]
    #             # a[i,j] = params['ad']*BS[j]**params['ab']*BS[i]**params['ac'] # attack rate of consumer j on prey i
    #             # h[i,j] = params['hd']*BS[j]**params['hb']*BS[i]**params['hc'] # handling time of consumer j of prey i
                
    #             a[i,j] = params['ad']*BS[j]**params['ab']*BS[i]**params['ac'] # attack rate of consumer j on prey i
    #             h[i,j] = params['hd']*BS[j]**params['hb']*BS[i]**params['hc'] # handling time of consumer j of prey i


    a = params['ad']*BS.reshape(Stot,1)**params['ab']*BS**params['ac']*FW["M"]
    h = params['hd']*BS.reshape(Stot,1)**params['hb']*BS**params['hc']*FW["M"]

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.scatter(np.divide(BS,BS.reshape(Stot,1)), a)
    
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.scatter(np.divide(BS,BS.reshape(Stot,1)), h)
    # plt.title("handling time")

    
    scatter = plt.scatter(coords['x'],coords['y'], c = np.arange(P), label = np.arange(P))
    plt.legend(*scatter.legend_elements(), title = "Patches")
    plt.title("Patch map")
    plt.show()

    # then calculate distances
    distances = get_distance(coords)
    
    ## maximum dispersal distance (Häussler et al. 2021)
    ## scales with body size for animals and uniformly drawn from uniform distribution for plants
    dmax = np.array([params["d0"]*BS[i]**params["eps"] if sum(M[:,i]) > 0 else np.random.uniform(0,0.5) for i in range(Stot)])
    
    # and create matrix of patch accessibility for each species depending on their starting point
    access = np.repeat(np.zeros(shape = (P-1, Stot)), P).reshape(P,P-1,Stot) # create empty array
    dd = np.repeat(np.zeros(shape = (P-1, Stot)), P).reshape(P,P-1,Stot)
    for patch in range(P):
        # neighbouring patches
        n = [j for j in range(P) if j!=patch] # index of neighbouring patches
        dn = distances[:,patch][n] # distance of neighbouring patches
        acc = np.array([dmax > j for j in dn])*1 # accessible patches for each species
        access[patch] = acc
        
        ddnz = np.array([i/dmax for i in dn])*acc # dispersal death while travelling from n to z (patch)
        dd[patch] = ddnz
        
    FW.update({'sp_ID':sp_ID, 'x':x,'r':r,'a':a,'h':h,'y0':y0,'K':K,"dmax":dmax,
               "dd":dd,"access":access,"distances":distances,"coords":coords})
    return(FW)
                    



from csv import writer
import time


def TwoPatchesPredationBS_ivp_Kernel(y0,q,P,S,FW,d,deltaR,harvesting,tfinal,tinit,s,extinct = 1e-14):
        
    # create an extinction event:
    def extinction(t,y,q,P,S,FW,d,deltaR,harvesting,s):
        
        # extinction
        if ((y <= extinct) & (y != 0)).any():
            
            index = np.where(((y <= extinct) & (y != 0)))
            print(y[index])
            print(index)
            np.savetxt(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Sp_extinct_event_{s}.csv",index)
            
            return 0 ## will terminate the ode solver
        
        # no extinction
        else: 
            return 1 ## solver can continue
                
    extinction.terminal = True # creates a termination event instead of just recordning the time step and biomass
    # extinction.direction = -1 # only species going down will trigger event, so invading species can invade
        
    
    def System(t,y,q,P,S,FW,d,deltaR,harvesting,s): # Function defining the system of differential equations
        
        a = FW['a'] # attack rate
        ht = FW['h'] # handling time
        x = FW['x'] # metabolic rate
        r = FW['r'] # growth rate
        K = FW['K'] + 0.00001 # just to prevent division by zero
        # dmax = FW["dmax"] # maximum dispersal distance 
        dd = FW["dd"]
        access = FW["access"]
        
        # empty population size
        dN = np.zeros(S*P).reshape(P,S)
        y = np.array(y).reshape(P,S)
        y[y < 0] = 0
        
        for p in range(P):
            # current patch
            N = y[p]  # previous population Np(t)
            dNp = np.zeros(S)  # initialising Np(t + 1)
            
            # neighbouring patches
            n = [j for j in range(P) if j!=p] # index of neighbouring patches
            # Nn = [y[j] for j in n]
            
            Nn = y[n,:] # biomass in neighbouting patches
            Nn = Nn*access[p] # filter species' biomass from patches that are within their maximum disp distance
            
            # =================================================================
            # (1) PREDATION
            # resources in rows, consumers in columns
            
            # Functionnal response: resource dependant feeding rate of consumer j on prey i influenced by j's k (other?) resources 
            # Fij = aij*Bi/(1+sum(aik*hik*Nk)) 
            # q = 1 # hill exponent (Sentis et al - between type II (hyperbolic q = 1) and type III (sigmoidal q = 2))
            # increased from 1.2 to 1.4 after suggestion of Benoit Gauzens for increasing persistence. Could also think about adding interference compeition.
            # who wrote ATNr package for allometric food webs
            
            F = a*(N.reshape(S,1)**(1 + q))
            low = 1 + np.matmul(np.transpose(a*ht), N**(1 + q)) 
            
            F = np.divide(F,low)
            
            predationGains = N*F.sum(axis = 0)*0.85 # xi*e*Ni*sum(Fik)
            predationLosses = np.matmul(N,F.T) # sum(xj*Nj*Fki)
            
            # =================================================================
            # (2) DISPERSAL 
            emigration = -d[p]*N # biomass that leaves the source patch (z)
            
            En = Nn*d[n,:] # emigrants leaving neighbouring patches 
            
            low_d = np.sum(1 - dd[n,:,:], axis = 0)
            Nim = np.sum(En * (1 -  dd[p]) * (1 -  dd[p])/low_d, axis = 0)
            
            # Nim = np.zeros(S) 
            # for ind in range(len(n)): # loop across neighbours 
            #     Nim+= (d[p]/(P-1))*Nn[ind]*connectivity[p,n[ind]] # biomass of each species in neighbouring patches that successfully disperse
            
            
            immigration = Nim # biomass not lost to the matrix during dispersal
            immigration[immigration<extinct] = 0
                  
            
            # print('immigration',immigration,'emigration', emigration)
            # print(N, 'Gains', predationGains, 'loss', predationLosses)
            
            # =================================================================
            
            # growth rate*logistic growth (>0 for plants only) - metabolic losses + predation gains (>0 for animals only) - predation losses + dispersal
            dNp = N*(r[p]*deltaR[p]*(1 - N/K) - x - harvesting[p]) + predationGains - predationLosses + immigration + emigration

                    
            # print(dNp)
            dNp[(N + dNp) < extinct] = 0 
            dN[p] = dNp
        # print(dN[0][33],dN[1][33],dN[2][33])
        return np.array(dN).reshape(P*S)
        
    
    # choose the start and end time
    t_span = (tinit,tfinal)
    # run solver
    return solve_ivp(fun = System, t_span = t_span, y0 = y0, args=(q,P,S,FW,d,deltaR,harvesting,s), 
                     method='LSODA', rtol = 1e-8, atol = 1e-8, events = extinction, t_eval = np.arange(tinit,tfinal,60*60*24*31)) # choose when extinctions and dispersal will be evaluated
    ### END FUNCTION


# =============================================================================
# run_dynamics()
#
# Runs simulations and re starts the dynamics when species go extinct after 
# setting their biomasses to zero and incrementing one timestep.
# It also checks if the simulations have stabilised (last 10% of biomasses have a cv < 1e-5) 
# and stops the simulations when that conditions is reached
# =============================================================================

def run_dynamics(y0,tinit,runtime,q,P,Stot,FW,disp,deltaR,harvesting,patch,landscape_seed,s,stage,sol_save={}):
    status = 10
    tstart = tinit
    stabilised = False
    count = 0
    
    start = time.time()    
    
    if min(deltaR) == max(deltaR):
        ty = 'Homogeneous'
    else:
        ty = 'Heterogeneous'

        
    while not stabilised:
        
        count+=1
        
        # print(tinit)
        sol = TwoPatchesPredationBS_ivp_Kernel(y0,q,P,Stot,FW,disp,deltaR,harvesting,tinit+runtime,tinit,s) # Call the model
        sol.y = sol.y.T
        
        print(tinit+runtime) # changed so that tmax extends with each extinction

        if tinit == tstart:
            sol_save = sol.copy()
            
        else:
            for i in ["y","t"]:
                sol_save[i] = np.concatenate((sol_save[i], sol[i])) 
                
        
        status = sol.status # update status of the solver (has it reach tfinal yet?)
        
        ## Checking if the dynamics have stablised
        if sol_save['y'].shape[0] > 1000: ## initial burn in period to let the dynamics unfold
            
            solY = sol_save['y']
            solT = sol_save['t']
            
            ### sliding window of variation of the mean biomass for each species
            mean_df = np.zeros((5,solY.shape[1])) ## initiate empty matrix to store results
            ind = 0 # counter to loop through the mean_df matrix
            for i in np.flip([0,1,2,3,4]): # we subset the last 25% timesteps into 5 equal length subsections [[75-80%],[80-85%],[85-90%],[90-95%],[95-100%]]
                
                thresh_high = (solT[-1] - (solT[-1] - solT[1])*(0.05*i)) # get higher time step boundary (tfinal - X% * tfinal)
                thresh_low = (solT[-1] - (solT[-1] - solT[1])*(0.05*(i + 1))) # get lower time step boundary (tfinal - (X + 1)% * tfinal)
        
                ## index of those time steps between low and high boundaries
                index = np.where((solT >= thresh_low) & (solT < thresh_high))[0]
                ## biomasses over those 10% time steps
                Bsub = solY[index]
                
                # save mean biomass per species across time
                mean_df[ind,:] = Bsub.mean(axis=0)
                ind+=1 # increment index
                
            # calculate coefficient of variation of each species's mean biomass 
            # across 25% of the simulation 
            cv = mean_df.std(axis=0)/mean_df.mean(axis=0)
           
            if (cv[~np.isnan(cv)] < 1e-2).all() : # if all the coefficient of variation are small enough we stop the simulation
                print('Stabilised')
                stabilised = True
                
                # plt.plot(mean_df)
                # plt.savefig('/lustrehome/home/s.lucie.thompson/Metacom/Figures/stabilisation.png', dpi = 400, bbox_inches = 'tight')
        
                break
                
        
        tinit = sol.t[-1].copy() + 1 # update the initial time step to the last time step of the latest run (keep the previous time step in and restart at t+1)
        
        
        # if the biomasses haven't stabilised:
        
        if ((status == 0) and (not stabilised)): # if it has reached tfinal (didn't end because of an extinction), then we extend the simulation
            
            print('extended')
            # tmax = tmax + (tmax - tstart)*0.10
            # y0 = sol.y[-1]
        
        
        if ((status == 1) and (not stabilised)):
            
            # get ID of species that went extinct and are to be set to zero Biomass
            ID = np.loadtxt(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Sp_extinct_event_{s}.csv")
            np.savetxt(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Sp_extinct_event_{s}.csv",[]) # erase information from file
    
            ID = ID.astype(int)
            print(ID)

            # update initial conditions, set to zero all extinct species      
            y0 = sol.y[-1].copy() # get latest vector of biomass
            y0[ID] = 0
            
        
        if count%200 == 0: # save progress every 200 extinctions
        
            sol_save_temp = sol_save.copy()
            sol_save_temp['y'] = sol_save_temp['y'][np.arange(0,sol_save_temp['y'].shape[0],10),:]
            sol_save_temp['t'] = sol_save_temp['t'][np.arange(0,sol_save_temp['t'].shape[0],10)]
            
            print('SAVING -- y = ',sol_save_temp['y'].shape,'t = ', sol_save_temp['t'].shape)
    
            ## save results
            sol_save_temp.update({'FW_new':FW, 'type':ty, "sim":s,"disp":disp,"harvesting":harvesting,"deltaR":deltaR,'q':q})
            np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/{ty}/sim{s}/temp_Patch{patch}_PopDynamics_{ty}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.npy',sol_save_temp, allow_pickle = True)
        
        ## simulation is taking too long we stop it - save it as a seperate file 
        ## need to investigate what is going on 
        if time.time() - start > 60*60*12:
            print(f'Took more than 12 hours - skipping sim {s}, patch {patch}, {ty}')
            
            ## save results
            sol_save_temp.update({'FW_new':FW, 'type':ty, "sim":s,"disp":disp,"harvesting":harvesting,"deltaR":deltaR,'q':q})
            np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/{ty}/sim{s}/NotStabilised_{stage}_landscape_{landscape_seed}_PatchRestored{patch}_PopDynamics_{ty}_sim{s}_{P}Patches_Stot100_C{int(C*100)}.npy',sol_save_temp, allow_pickle = True)

            return 'Did not stabilise after 12 hours'
        
    return sol_save


# =============================================================================
# reduce_FW()
#
# subset the food web to only species that are present in the landscape
# =============================================================================
 
def reduce_FW(FW, y0, P, disp):
    
    Stot = FW['Stot']
    
    # subset only species that are present in the landscape
    boo = np.repeat(False, Stot)
    for i in range(P):
        boo = ((boo) | (y0[i]!=0))
    # get their index from the original food web
    ID_original = FW['sp_ID'][boo]  
    present = np.where(boo)[0]
    # realised species richness    
    Stot_new = len(present)
    # updated initial conditions
    y0_new = y0.reshape(Stot*P,)[np.tile(present, P) + np.repeat(np.arange(P)*Stot, Stot_new)].reshape(P,Stot_new)
    # check - should be none
    # print(np.where((y0_new[0]==0) & (y0_new[1]==0) & (y0_new[2]==0))[0])
    
    ## subset the food webs and all its parameters
    FW_new = FW.copy()
    FW_new['Stot'] = Stot_new
    FW_new['M'] = FW_new['M'][np.ix_(present,present)]
    FW_new['a'] = FW_new['a'][np.ix_(present,present)]
    FW_new['h'] = FW_new['h'][np.ix_(present,present)]
    FW_new['r'] = FW_new['r'].reshape(Stot*P,)[np.tile(present, P) + np.repeat(np.arange(P)*Stot, Stot_new)].reshape(P,Stot_new)
    FW_new['x'] = FW_new['x'][present]
    FW_new['K'] = FW_new['K'][present]
    FW_new['TL'] = FW_new['TL'][present]
    
    access = np.repeat(np.zeros(shape = (P-1, Stot_new)), P).reshape(P,P-1,Stot_new)
    dd = np.repeat(np.zeros(shape = (P-1, Stot_new)), P).reshape(P,P-1,Stot_new)
    for i in range(P):
        access[i] = FW_new['access'][i][:,present]
        dd[i] = FW_new['dd'][i][:,present]
    
    FW_new['access'] = access
    FW_new['dd'] = dd
    
    # keep in memory which species we kept from the regional food web
    FW_new['sp_ID'] = ID_original
    FW_new['sp_ID_'] = present # sp ID for the subsetted food web 
    FW_new['y0'] = y0_new
    disp_new = disp.reshape(Stot*P,)[np.tile(present, P) + np.repeat(np.arange(P)*Stot, Stot_new)].reshape(P,Stot_new)
    
    return Stot_new, FW_new, disp_new





# %% initial run homogeneous
f = "/lustrehome/home/s.lucie.thompson/Metacom/Init_test/StableFoodWebs_55persist_Stot100_C10_t10000000000000.npy"
stableFW = np.load(f, allow_pickle=True).item()


Stot = 100 # initial pool of species
P = 15 # number of patches
C = 0.1 # connectance
tmax = 10**12 # number of time steps

# FW = nicheNetwork(Stot, C)

# S = np.repeat(Stot,P) # local pool sizes (all patches are full for now)
S = np.repeat(round(Stot*1/3),P) # initiate with 50 patches

parser = ArgumentParser()
parser.add_argument('SimNb')
args = parser.parse_args()
s = int(args.SimNb)
print(args, s, flush=True)

# s = 0

import os 

## create file to load extinction events
path = f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Sp_extinct_event_{s}.csv"
if not os.path.exists(path):
    file = open(f"/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Sp_extinct_event_{s}.csv","x") # create a file
    file.close()
    
    
print('simulation',s, flush=True)

k = [k for k in stableFW.keys()]
k = k[s]
FW = stableFW[k]


## create three patches distant enough that all species can't cross all the way through
extentx = [0,0.4]
radius_max = 0.05

## surface of the circle
((radius_max**2)*np.pi)/(0.5*0.5)
## surface of the landscape

stage = 'Init'

for seed_index in [0,1,3,2,4,5]:

    seed = np.arange(P*seed_index, P*(seed_index + 1))
    coords = create_landscape(P = P, extent = extentx, radius_max = radius_max, nb_center = 5, seed = seed)
    
    
    FW = getSimParams(FW, S, P, coords, extentx)
    
    ## calculate distance between patches
    dist = FW['distances']
    np.sum(dist, axis = 0)
    np.median(dist) ## should be close to np.median(FW['dmax'])
    np.median(dist[0:5,0:5])
    
    plt.xscale("log")
    plt.scatter(FW["BS"], FW["dmax"])
    plt.ylabel("dispersal distance")
    plt.xlabel("Logged body size")
    plt.show()
    
    # dispersal rate
    d = 1e-8
    disp = np.repeat(d, Stot*P).reshape(P,Stot)
    
    q = 0.1 # hill exponent - type II functionnal response (chosen following Ryser et al 2021)
    
    #### (1) Homogeneous landscape
    deltaR = np.repeat(0.5,P)
    
    Stot_new, FW_new, disp_new = reduce_FW(FW, FW['y0'], P, disp)
    # no havresting for the 'warm up' period
    harvesting = np.zeros(shape = (P, Stot_new))
    
    #### run the initial transtional state for the food web (no disturbance yet)
    
    B_init = FW_new['y0']
    
    import os
    
    path_files = f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/init_files_15Patches_narrow.csv'
    init_files = np.loadtxt(path_files, delimiter = ",", dtype = str)
    
    file_name = f'InitialPopDynamics_seed{seed_index}_narrow_homogeneous_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.pkl'
    NotStabilised_filename = f'NotStabilised_{stage}_landscape_{seed_index}_PatchRestored-1_PopDynamics_homogeneous_sim{s}_{P}Patches_Stot100_C{int(C*100)}.npy'
    
    if file_name not in init_files and NotStabilised_filename not in init_files:
    
        print('Homogeneous - initial run',s)
        
        start = time.time()    
        sol_homogeneous = run_dynamics(FW_new['y0'].reshape(Stot_new*P,),0,tmax,q,P,Stot_new,FW_new,disp_new,deltaR,harvesting,-1,seed_index,s,'Init')
        stop = time.time() 
        sim_duration = stop - start
        print(sim_duration)
        
        if sol_homogeneous != 'Did not stabilise after 12 hours':
        
            Bf_homogeneous = np.zeros(shape = (P,Stot_new))
            res_sim_homogeneous = pd.DataFrame()
            for patch in range(P):
                
                p_index = patch + 1
                
                # sol_ivp_k1["y"][sol_ivp_k1["y"]<1e-20] = 0         
                solT = sol_homogeneous['t']
                solY = sol_homogeneous['y']
                ind = patch + 1
                
                Bf_homogeneous[patch] = solY[-1,range(Stot_new*ind-Stot_new,Stot_new*ind)]
                
                
                    # plot pop dynamics for all species
            #     plt.subplot(1,3, ind)
            #     plt.tight_layout()
                
            #     plt.loglog(solT[np.arange(0,solT.shape[0],10)], 
            #                solY[np.ix_(np.arange(0,solT.shape[0],10),range(Stot_new*ind-Stot_new,Stot_new*ind))])
            # plt.title("Homogeneous")
            # plt.show()
                    
            print('shape raw',sol_homogeneous['y'].shape)
            print('shape subset 10%',sol_homogeneous['y'][np.arange(0,sol_homogeneous['y'].shape[0],10),:].shape)
            
            sol_homogeneous['y'] = sol_homogeneous['y'][np.arange(0,sol_homogeneous['y'].shape[0],10),:]
            sol_homogeneous['t'] = sol_homogeneous['t'][np.arange(0,sol_homogeneous['t'].shape[0],10)]
        
            ## save results
            sol_homogeneous.update({'FW_new':FW_new, 
                                    'type':'homogeneous', 
                                    'FW':FW, 
                                    "sim":s,
                                    'FW_ID':k,
                                    "FW_file":f,
                                    "disp":disp,
                                    "harvesting":harvesting,
                                    "deltaR":deltaR,
                                    'tstart':0, 
                                    'tmax':tmax,
                                    'q':q, 
                                    'sim_duration':sim_duration,
                                    'lanscape_seed':seed_index,
                                    'subset':1/10
                                    })
            
            with open(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/InitialPopDynamics_seed{seed_index}_narrow_homogeneous_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.pkl', 'wb') as file:  # open a text file
                pickle.dump(sol_homogeneous, file, protocol=4) # serialize the list
            file.close()
            
            # add file to list of files that ran
            init_files = np.append(init_files, file_name)
            np.savetxt(path_files, init_files, delimiter=',', fmt='%s')
            
      
        
    
    else:
        
        # sol_homogeneous = np.load(path, allow_pickle = True).item()
        
        with open(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/InitialPopDynamics_seed{seed_index}_narrow_homogeneous_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{tmax}r.pkl', 'rb') as file: 
            sol_homogeneous = pickle.load(file)  
        file.close()
        
        Bf_homogeneous = np.zeros(shape = (P,Stot_new))
        for patch in range(P):
            
            p_index = patch + 1
            
            # sol_ivp_k1["y"][sol_ivp_k1["y"]<1e-20] = 0         
            solT = sol_homogeneous['t']
            solY = sol_homogeneous['y']
            ind = patch + 1
            
            Bf_homogeneous[patch] = solY[-1,range(Stot_new*ind-Stot_new,Stot_new*ind)]
            
        ###############################################################################
    
    # %% Starting landscape recovery
        
    '''
    
      We iteratively improve patches one by one and to test whether cluster size
      increases restoration outcome.
     
      We do the same by improving 'random' non-clustered patches to compare. 
    
    '''
        
    
    Binit_restored = Bf_homogeneous.copy()
    tstart = sol_homogeneous["t"][-1].copy()
    tinit = tstart.copy()
    runtime = 1e11
    disturbance = 5e-6
    ty = 'Heterogeneous'
    
    print('Heterogeneous - restoration',s, flush=True)
    
    
    
    # Get reduced space
    FW_restored_new = FW_new.copy()
    FW_restored_new['y0'] = Binit_restored
    
    
    path_files = f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/control_files_15Patches_narrow.csv'
    control_files = np.loadtxt(path_files, delimiter = ",", dtype = str)
    
    control_file_name = f'CONTROL-PopDynamics_homogeneous_seed{seed_index}_narrow_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.pkl'
    
    
    if control_file_name not in control_files:
        print('Control')
        
        # check if the species isn't extinct
        y0 = FW_restored_new['y0'].reshape(P*Stot_new)    
        
        harvesting = np.zeros(shape = (P, Stot_new))
        
        ## in the control - the invasive species should not be able to invade 
        start = time.time()   
        sol_control = run_dynamics(y0, tstart, tinit + runtime, q, P, Stot_new, FW_restored_new, disp_new, deltaR, harvesting, -1,seed_index, s, 'control')
        stop = time.time()   
        sim_duration = stop - start
        print(sim_duration)
            
        sol_control.update({'FW':FW, 
                            'type':'homogeneous', 
                            'FW_new':FW_restored_new,
                            'Stot_new':Stot_new, 
                            "sim":s,
                            'FW_ID':k,
                            "FW_file":f,
                            "disp":disp_new,
                            "harvesting":harvesting,
                            "deltaR":deltaR,
                            'tstart':tstart, 
                            'runtime':runtime,
                            'q':q,
                            'sim_duration':sim_duration,
                            'lanscape_seed':seed_index
                            })
        
        with open(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/CONTROL-PopDynamics_homogeneous_seed{seed_index}_narrow_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.pkl' , 'wb') as file:  # open a text file
            pickle.dump(sol_control, file, protocol=4) # serialize the list
        file.close()
        
        # add file to list of files that ran
        # control_files = np.append(control_files, control_file_name)
        # np.savetxt(path_files, control_files, delimiter=',', fmt='%s')
            
        # np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/CONTROL-PopDynamics_homogeneous_seed{seed_index}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.npy',sol_control, allow_pickle = True)
    
    
    # mean biomass of plants in the landscape:
    mean_biomass = np.mean(FW_restored_new['y0'])
    
    # to allow for re-invasion of species from the regional pool, we set extinct species' biomasses to 1/100 th of their extant equivalent
    
    # invaded patch is the closets to the upper left corner
    distance_left = np.array(((coords['x'] - extentx[0])**2 + (coords['y'] - extentx[1])**2)**(1/2))
    patch_to_invade = np.argmin(distance_left)
    
    # no invaders except in the top left corner patch
    FW_restored_new['invaders'] = np.repeat(False, Stot*P).reshape(P,Stot)
    # invaders are all the species that were absent from this top left corner patch
    FW_restored_new['invaders'][patch_to_invade] = FW_restored_new['y0'][patch_to_invade] == 0
    FW_restored_new['mean_invader_biomass'] = mean_biomass
    
    # allow invasion only in outside patches
    temp = FW_restored_new['y0'][patch_to_invade]
    temp[np.where(temp == 0)] = mean_biomass/100
    FW_restored_new['y0'][patch_to_invade] = temp
    
    control_file_name = f'CONTROL-invasion-CornerPatch-PopDynamics_homogeneous_seed{seed_index}_narrow_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.pkl'
        
    if control_file_name not in control_files:
        
        print('Control - invasion')
        
        # check if the species isn't extinct
        y0 = FW_restored_new['y0'].reshape(P*Stot_new)     # biomass with invaders
        
        harvesting = np.zeros(shape = (P, Stot_new))
        
        ## in the control - the invasive species should not be able to invade 
        start = time.time()   
        sol_control = run_dynamics(y0, tstart, tinit + runtime, q, P, Stot_new, FW_restored_new, disp_new, deltaR, harvesting, -1,seed_index, s, 'control_invasion')
        stop = time.time() 
        sim_duration = stop - start
        print(sim_duration)
            
        sol_control.update({'FW':FW, ## initial food web before any reduction
                            'B_init': B_init, ## very first initialisation of biomasses
                            'B_final_homogeneous': Bf_homogeneous, ## biomass after first initial run on homogeneous landscape (pre-restoration / invasion)
                            'type':'homogeneous', ## type of landscape
                            'FW_new':FW_restored_new,
                            'Stot_new':Stot_new, 
                            "sim":s,
                            'FW_ID':k,
                            "FW_file":f,
                            "disp":disp_new,
                            "harvesting":harvesting,
                            "deltaR":deltaR,
                            'tstart':tstart, 
                            'runtime':runtime,
                            'q':q,
                            'sim_duration':sim_duration,
                            'lanscape_seed':seed_index
                            })
        
        with open(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/CONTROL-invasion-CornerPatch-PopDynamics_homogeneous_seed{seed_index}_narrow_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.pkl', 'wb') as file:  # open a text file
            pickle.dump(sol_control, file, protocol=4) # serialize the list
        file.close()
        
        # add file to list of files that ran
        control_files = np.append(control_files, control_file_name)
        np.savetxt(path_files, control_files, delimiter=',', fmt='%s')
        
        # np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Homogeneous/sim{s}/CONTROL-invasion-PopDynamics_homogeneous_seed{seed_index}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.npy',sol_control, allow_pickle = True)
    
    
    ## scattering randomly patches to improve
    # patches to improve
    list_patches_to_restore = [coords[coords['position'] == 'center']['Patch']]
    list_restoration_types = ['clustered']  
    list_seeds = [None]
    for seed in np.arange(5): ## draw 5 random patches to improve across the landscape
        np.random.seed(seed)
        list_patches_to_restore = list_patches_to_restore + [np.random.choice(np.arange(15), 5, replace=False)] # random number between [0,15)
        list_restoration_types = list_restoration_types + ['scattered'] 
        list_seeds = list_seeds + [seed]
    
    y0 = FW_restored_new['y0'].reshape(P*Stot_new)
    harvesting = np.zeros(shape = (P, Stot_new))  
    
    stage = 'restoration'
    
    for patches_to_restore, restoration_type, restoration_seed in zip(list_patches_to_restore, list_restoration_types, list_seeds):
        
        patch_to_improve = [] ## initialise list of patches to improve
        
        print(patches_to_restore, restoration_type)
        
        for patch in patches_to_restore:
            
            # start with one patch, then add the 4 others one by one
            patch_to_improve = patch_to_improve + [patch]
                    
        
            path_files = f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/improved_files_15Patches_narrow.csv'
            improved_files = np.loadtxt(path_files, delimiter = ",", dtype = str)
            
            improved_file_name = f'PopDynamics_heterogeneous-invasion-CornerPatch_narrow_seed{seed_index}-restoration_seed{restoration_seed}_{restoration_type}_patchImproved{patch}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.pkl'
            path_notStabilised = f'NotStabilised_{stage}_landscape_{seed_index}_PatchRestored{patch}_PopDynamics_{ty}_sim{s}_{P}Patches_Stot100_C{int(C*100)}.npy'

            
            if improved_file_name not in improved_files and path_notStabilised not in improved_files:    
                
                print('patch improved:', patch_to_improve, flush=True)
                
                deltaR = np.repeat(0.5,P)
                deltaR[patch_to_improve] = 1.5
                    
                # run dynamics from equilibrium Bf_disturbed
                # Bf_disturbed = Bf_disturbed.reshape(P*Stot_new,)
                
                start = time.time()   
                sol_heterogeneous_restored = run_dynamics(y0, tstart, tinit + runtime, q, P, Stot_new, FW_restored_new, disp_new, deltaR, harvesting, patch, seed_index, s, 'restoration')
                stop = time.time() 
                sim_duration = stop - start
                print(sim_duration)
                    
                if sol_heterogeneous_restored != 'Did not stabilise after 12 hours':
                
                    sol_heterogeneous_restored.update({'FW':FW, ## initial food web before any reduction
                                                       'type':ty, ## type of landscape
                                                       'restoration_type':restoration_type, ## whether high quality patches are clustered or not
                                                       'restored_patches':patches_to_restore,
                                                       'restored_patches_seed':restoration_seed,
                                                       'lanscape_seed':seed_index,
                                                       
                                                       'B_init': B_init, ## very first initialisation of biomasses
                                                       'B_final_homogeneous': Bf_homogeneous, ## biomass after first initial run on homogeneous landscape (pre-restoration / invasion)
                                                       'FW_new':FW_restored_new, ## Reduced food web with only regionally extant species after intial run - used for invasion/restoration experiment
                                                       'Stot_new':Stot_new, ## regional species richness (dimensions of FW_new)
                                                       "sim":s, ## Food web ID number
                                                       'FW_ID':k, ## 
                                                       "FW_file":f, ## food web file
                                                       "disp":disp_new, ## species maximal dispersal
                                                       "harvesting":harvesting, ## species harvesting 
                                                       "deltaR":deltaR, ## patch quality
                                                       'tstart':tstart, ## start time of restoration 
                                                       'runtime':runtime, ## maximum runtime allowed
                                                       'q':q, ## hill number
                                                       'patch_to_improve':patch_to_improve, ## ID of patch(es) with higher quality
                                                       'sim_duration':sim_duration,
                                                       'subset':1/10
                                                       })
                    # np.save(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Heterogeneous/sim{s}/DisturbedP{patch}PopDynamics_heterogeneous_seed{seed_index}_patchImproved{patch}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}_disturbance{disturbance}_{sp}SpDisturbed.npy',sol_heterogeneous_restored, allow_pickle = True)
                    
                    print('Types ',[type(sol_heterogeneous_restored[k]) for k in sol_heterogeneous_restored.keys()], flush=True)
                    print('keys ',sol_heterogeneous_restored.keys(), flush=True)
                    print(sol_heterogeneous_restored, flush=True)
                    
                    sol_heterogeneous_restored['y'] = sol_heterogeneous_restored['y'][np.arange(0,sol_heterogeneous_restored['y'].shape[0],10),:]
                    sol_heterogeneous_restored['t'] = sol_heterogeneous_restored['t'][np.arange(0,sol_heterogeneous_restored['t'].shape[0],10)]

                    with open(f'/lustrehome/home/s.lucie.thompson/Metacom/{P}Patches/Heterogeneous/sim{s}/PopDynamics_heterogeneous-invasion-CornerPatch_narrow_seed{seed_index}-restoration_seed{restoration_seed}_{restoration_type}_patchImproved{patch}_sim{s}_{P}Patches_Stot{Stot}_C{int(C*100)}_t{runtime}.pkl', 'wb') as file:  # open a text file
                        pickle.dump(sol_heterogeneous_restored, file, protocol=4) # serialize the list
                    file.close()
                    
                    improved_files = np.append(improved_files, improved_file_name)
                    np.savetxt(path_files, improved_files, delimiter=',', fmt='%s')
                    # save final biomass density after perturbation and plot dynamics
