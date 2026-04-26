
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:40:50 2024

@author: mariapau
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy as sp
import seaborn as sb
from scipy import stats
import pickle


#######################################################################
## Definicion de funciones  ###########################################
#######################################################################

def leo_archivos(ritmo, threshold, Pts_idx):
    
    os.chdir('./')
    print(os.getcwd())
    
    Pt_dict = pd.read_csv('/Users/mariapau/Conicet/Grafos_IAM /Pt_dict.csv')
    del Pt_dict['Unnamed: 0']
    
    print('Patient dict: ', Pt_dict.keys())
    
    
    grado_A = []
    grado_C = []
    grado_F = []    
    AC_A = []
    AC_C = []
    AC_F = []    
    lclus_dist_A = []
    lclus_dist_C = []
    lclus_dist_F = []    
    gclus_A = []
    gclus_C = []
    gclus_F = []
    entr_A = []
    entr_C = []
    entr_F = []    
    entr_sig_A = []
    entr_sig_C = []
    entr_sig_F = []
    densidad_A = []
    densidad_C = []
    densidad_F = []
    LE_A = []
    LE_C = []
    LE_F = []
    EC_A = []
    EC_C = []
    EC_F = []
    SG_A = []
    SG_C = []
    SG_F = []
    SR_A = []
    SR_C = []
    SR_F = []
    dist_A = []
    dist_C = []
    dist_F = []
    Pot_A = []
    Pot_C = []
    Pot_F = []       
    age_A = []
    age_C = []
    age_F = []       
    MMSE_A = []
    MMSE_C = []
    MMSE_F = []       
    
    nombre_conect = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/density.pkl'
    nombre_grado = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/degree_distribution.pkl'
    nombre_gclus = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) +'/global_clustering.pkl'
    nombre_lclus = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/local_clustering.pkl'
    nombre_dist = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) +  '/dist_T4_T3_T5_T6.pkl'
    nombre_entr = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/entropy.pkl'
    nombre_entr_sig = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/entropy_sig.pkl'
    nombre_LE = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) +  '/LE.pkl'
    nombre_SG = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) +  '/SG.pkl'
    nombre_SR = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/SR.pkl'
    nombre_EC = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/EC.pkl'
    nombre_AC = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) + '/AC.pkl'
    nombre_Pot = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/' + str(threshold) +'/Power.pkl'
    
    with open(nombre_conect, 'rb') as f:
        conect = pickle.load(f) # deserialize using load()
    with open(nombre_grado, 'rb') as f:
        grado = pickle.load(f) # deserialize using load()
    with open(nombre_dist, 'rb') as f:
        dist = pickle.load(f) # deserialize using load()
    with open(nombre_entr, 'rb') as f:
        entropy = pickle.load(f) # deserialize using load()
    with open(nombre_entr_sig, 'rb') as f:
        entr_sig = pickle.load(f) # deserialize using load()
    with open(nombre_LE, 'rb') as f:
        LE = pickle.load(f) # deserialize using load()
    with open(nombre_lclus, 'rb') as f:
        lclus_distr = pickle.load(f) # deserialize using load()
    with open(nombre_gclus, 'rb') as f:
        gclus = pickle.load(f) # deserialize using load()
    with open(nombre_SG, 'rb') as f:
        SG = pickle.load(f) # deserialize using load()
    with open(nombre_SR, 'rb') as f:
        SR = pickle.load(f) # deserialize using load()    
    with open(nombre_EC, 'rb') as f:
        EC = pickle.load(f) # deserialize using load()
    with open(nombre_AC, 'rb') as f:
        AC = pickle.load(f) # deserialize using load()
    with open(nombre_Pot, 'rb') as f:
        pot = pickle.load(f) # deserialize using load()
               
    
    #Construyo los histogramas de autovalores y grado de cada clase 

    for z,i in enumerate(range(Pts_idx[0][0], 4*Pts_idx[0][1])):
        print(i)
        dist_A.append(dist.iloc[z])
        grado_A.append(grado.iloc[z])
        lclus_dist_A.append(lclus_distr.iloc[z])
        entr_A.append(entropy.iloc[z][0])
        entr_sig_A.append(entr_sig.iloc[z][0])
        densidad_A.append(conect.iloc[z][0])
        gclus_A.append(gclus.iloc[z])
        SG_A.append(SG.iloc[z])
        SR_A.append(SR.iloc[z])
        EC_A.append(EC.iloc[z])
        AC_A.append(AC.iloc[z])
        LE_A.append(LE.iloc[z])
        Pot_A.append(pot.iloc[z])
        
    grado_A = np.array(grado_A)
    grado_A = grado_A.reshape((1,grado_A.size))
    dist_A = np.array(dist_A)
    loc_clus_A = np.array(lclus_dist_A)
    loc_clus_A = loc_clus_A.reshape((1,loc_clus_A.size))
    Pot_A = np.array(Pot_A)
    SR_A = np.array(SR_A)
    LE_A = np.array(LE_A)
    SG_A = np.array(SG_A)
    AC_A = np.array(AC_A)
    gclus_A = np.array(gclus_A)
    EC_A = np.array(EC_A)
    EC_A = EC_A.reshape((1,EC_A.size))
    entr_A = np.array(entr_A)
    entr_sig_A = np.array(entr_sig_A)

    for k,ii in enumerate(range(4*Pts_idx[1][0], 4*Pts_idx[1][1])):
        kk = z + k + 1
        print(ii)
        dist_C.append(dist.iloc[kk])
        grado_C.append(grado.iloc[kk])
        lclus_dist_C.append(lclus_distr.iloc[kk])
        entr_C.append(entropy.iloc[kk][0])
        entr_sig_C.append(entr_sig.iloc[kk][0])
        LE_C.append(LE.iloc[kk][0])
        densidad_C.append(conect.iloc[kk][0])        
        gclus_C.append(gclus.iloc[kk])
        SG_C.append(SG.iloc[kk])
        SR_C.append(SR.iloc[kk])
        EC_C.append(EC.iloc[kk])
        AC_C.append(AC.iloc[kk])
        Pot_C.append(pot.iloc[kk])

    grado_C = np.array(grado_C)
    grado_C = grado_C.reshape((1,grado_C.size))
    dist_C = np.array(dist_C)
    loc_clus_C = np.array(lclus_dist_C)
    loc_clus_C = loc_clus_C.reshape((1,loc_clus_C.size))
    gclus_C = np.array(gclus_C)
    Pot_C = np.array(Pot_C)
    SR_C = np.array(SR_C)
    SG_C = np.array(SG_C) 
    LE_C = np.array(LE_C) 
    AC_C = np.array(AC_C) 
    EC_C = np.array(EC_C)
    EC_C = EC_C.reshape((1,EC_C.size))
    entr_C = np.array(entr_C)
    entr_sig_C = np.array(entr_sig_C)
    
    for j,ii in enumerate(range(4*Pts_idx[2][0], 4*Pts_idx[2][1])):
        print(ii)
        jj = j + kk + 1
        dist_F.append(dist.iloc[jj])
        grado_F.append(grado.iloc[jj])
        lclus_dist_F.append(lclus_distr.iloc[jj])
        entr_F.append(entropy.iloc[jj][0])
        entr_sig_F.append(entr_sig.iloc[jj][0])
        gclus_F.append(gclus.iloc[jj])
        densidad_F.append(conect.iloc[jj][0])
        SG_F.append(SG.iloc[jj])
        SR_F.append(SR.iloc[jj]) 
        LE_F.append(LE.iloc[jj]) 
        EC_F.append(EC.iloc[jj])
        AC_F.append(AC.iloc[jj])
        Pot_F.append(pot.iloc[jj])
            
    dist_F = np.array(dist_F)
    grado_F = np.array(grado_F)
    grado_F = grado_F.reshape((1,grado_F.size))
    loc_clus_F = np.array(lclus_dist_F)
    loc_clus_F = loc_clus_F.reshape((1,loc_clus_F.size))
    gclus_F = np.array(gclus_F)
    Pot_F = np.array(Pot_F)
    SR_F = np.array(SR_F)
    LE_F = np.array(LE_F)
    SG_F = np.array(SG_F)    
    AC_F = np.array(AC_F)
    EC_F = np.array(EC_F)
    EC_F = EC_F.reshape((1,EC_F.size))
    entr_F = np.array(entr_F)
    entr_sig_F = np.array(entr_sig_F)
    
    #armos dataframes con las tres clases
    dist =  {'A': dist_A.flatten(), 'C': dist_C.flatten(), 'F': dist_F.flatten()}
    grado = {'A': grado_A[0], 'C': grado_C[0], 'F': grado_F[0]}
    loc_clus = {'A': loc_clus_A[0], 'C': loc_clus_C[0], 'F': loc_clus_F[0]}
    g_clus = {'A': gclus_A, 'C': gclus_C, 'F': gclus_F}
    entr = {'A': entr_A, 'C': entr_C, 'F': entr_F}
    entr_sig = {'A': entr_sig_A, 'C': entr_sig_C, 'F': entr_sig_F}
    dens = {'A': densidad_A, 'C': densidad_C, 'F': densidad_F}
    SG = {'A': SG_A, 'C': SG_C, 'F': SG_F}
    SR = {'A': SR_A, 'C': SR_C, 'F': SR_F}
    EC = {'A': EC_A, 'C': EC_C, 'F': EC_F}
    AC = {'A': AC_A, 'C': AC_C, 'F': AC_F}
    LE = {'A': LE_A, 'C': LE_C, 'F': LE_F}
    POT = {'A': Pot_A, 'C': Pot_C, 'F': Pot_F}

    return  AC, dist, grado, loc_clus, g_clus, entr, dens, LE, SG, SR, EC, POT, entr_sig 
 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
def grafico_distribuciones(A,C,F, titulo,chs_eeg):
    # Histograma de autovalores
    canales_str = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    chs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    if chs_eeg:
        xticklabels = [canales_str[j] for j in chs]
    else:
        xticklabels = chs
    """
    fig, axs = plt.subplots(3,1, figsize = (12,6))
    axs[0].stairs(A[0], A[1], color='k')
    axs[0].set_title(titulo)
    axs[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[0].set_xticklabels(xticklabels)
    axs[1].stairs(C[0], C[1], color='k')
    axs[1].set_title(titulo)
    axs[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[1].set_xticklabels(xticklabels)
    axs[2].stairs(F[0], F[1], color='k')
    axs[2].set_title(titulo)
    plt.subplots_adjust(hspace=1)  # Cambia el valor para más o menos espacio
    axs[2].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[2].set_xticklabels(xticklabels)
    plt.show()
    """
    x = -0.2
    As = np.array(A[0])
    Cs = np.array(C[0])
    Fs = np.array(F[0])
    max_scale = max([max(As), max(Cs), max(Fs)])
    As = As/max_scale
    Cs = Cs/max_scale
    Fs = Fs/max_scale
    
    plt.figure(figsize = (12,6))
    plt.title(titulo)
    plt.stairs(As, A[1] + x, color='k', label = 'A')
    plt.stairs(Cs, C[1] + 2*x, color='r', label = 'C')
    plt.stairs(Fs, F[1] + 3*x, color='c', label = 'F')
    #max_scale = np.max([np.max(A[1]),np.max(C[1]), np.max(F[1])])
    plt.ylim([0, 1.1])
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], xticklabels)
    plt.legend()
    plt.show()
    
    
######################################################################
def grafico_cajas(A,C,F, titulo):   
    #Elimino nans 
    idx = np.isnan(A)
    A = np.delete(A, idx)    
    idx = np.isnan(C)
    C = np.delete(C, idx)    
    idx = np.isnan(F)
    F = np.delete(F, idx)    
    #Grafico
    plt.figure(figsize=(8,4))
    plt.boxplot([A, C, F])
    xticklabels = ['A','C','F']    
    plt.xticks([1,2,3], xticklabels)
    plt.title(titulo)
    #plt.savefig(nombre, dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
######################################################################  
