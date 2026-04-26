
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
def ajuste_mmse(x,y):


    # Realizar el ajuste lineal
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #slope, intercept = np.polyfit(x, y, 1)
    
    # Calcular los valores ajustados
    y_fit = slope * x + intercept
    
    # Mostrar los resultados del ajuste
    print(f'Pendiente: {slope}')
    print(f'Intersección: {intercept}')
    print(f'Valor de R cuadrado: {r_value**2}')
    
    # Graficar los datos y la línea ajustada
    plt.figure()
    plt.scatter(x, y, label='Datos')
    plt.plot(x, y_fit, color='red', label='Ajuste lineal')
    plt.xlabel('')
    plt.ylabel('y')
    plt.title('Ajuste lineal de dos variables')
    plt.legend()
    plt.show()
######################################################################

def analisis_grafos(ritmo, threshold, Pts_idx, estac):
    
    os.chdir('./')
    print(os.getcwd())
    
    Pt_dict = pd.read_csv('/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /Pt_dict.csv')
    del Pt_dict['Unnamed: 0']
    
    print('Patient dict: ', Pt_dict.keys())
    print('Ritmo: ', ritmo, 'Threshold: ', threshold, 'estac: ' , estac)
    
    
    avals_dist_A = []
    avals_dist_C = []
    avals_dist_F = []
    avals_A = []
    avals_C = []
    avals_F = []
    aval_nonulo_A = []
    aval_nonulo_C = []
    aval_nonulo_F = []
    grado_A = []
    grado_C = []
    grado_F = []    
    Traza_A = []
    Traza_C = []
    Traza_F = []
    Traza_A_std = []
    Traza_C_std = []
    Traza_F_std = []
    lclus_dist_A = []
    lclus_dist_C = []
    lclus_dist_F = []
    entr_A = []
    entr_C = []
    entr_F = []
    densidad_A = []
    densidad_C = []
    densidad_F = []
    LE_A = []
    LE_C = []
    LE_F = []
    dist_A = []
    dist_C = []
    dist_F = []
    

    
    canales_str = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    
# Leo avals y conect
    nombre_avals =  '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) + '/avals.pkl'
    nombre_conect = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/conectancia.pkl'
    nombre_transit = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/transitividad.pkl'
    nombre_grado = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/grado.pkl'
    nombre_lclus = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/lclus.pkl'
    nombre_dist = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/dist.pkl'
    nombre_entr = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/entropy.pkl'
    nombre_LE = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/LE.pkl'
    nombre_EC = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/EC.pkl'
    nombre_SR = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/SR.pkl'
    nombre_SG = '/Users/mariapau/Conicet/Becarios/Camila/Grafos_IAM /' + ritmo + '/' + str(threshold) + '_' + str(estac) +  '/SG.pkl'
    
    with open(nombre_avals, 'rb') as f:
        avals = pickle.load(f) # deserialize using load()
    with open(nombre_conect, 'rb') as f:
        conect = pickle.load(f) # deserialize using load()
    with open(nombre_grado, 'rb') as f:
        grado = pickle.load(f) # deserialize using load()
    with open(nombre_dist, 'rb') as f:
        dist = pickle.load(f) # deserialize using load()
    with open(nombre_entr, 'rb') as f:
        entropy = pickle.load(f) # deserialize using load()
    with open(nombre_LE, 'rb') as f:
        LE = pickle.load(f) # deserialize using load()    
    with open(nombre_EC, 'rb') as f:
        EC = pickle.load(f) # deserialize using load()
    with open(nombre_SR, 'rb') as f:
        SR = pickle.load(f) # deserialize using load()
    with open(nombre_SG, 'rb') as f:
        SG = pickle.load(f) # deserialize using load()
    with open(nombre_lclus, 'rb') as f:
        lclus_distr = pickle.load(f) # deserialize using load()
            
   
    
    #Construyo los histogramas de autovalores y grado de cada clase 

    for z,i in enumerate(range(Pts_idx[0][0], Pts_idx[0][1])):
        print(i)
        avals_dist_A.append(avals.iloc[z])  
        avals_A.append(avals.iloc[z])
        dist_A.append(dist.iloc[z])
        idx_2 = np.where(np.array(avals_A[z]) > 0.00001)[0]
        if idx_2.any():
            x = avals_A[z][idx_2[0]]   
        else:
            x = np.nan
        aval_nonulo_A.append(x)
        grado_A.append(grado.iloc[z])
        lclus_dist_A.append(lclus_distr.iloc[z])
        Traza_A.append(np.sum(avals.iloc[z])) 
        Traza_A_std.append(np.std(avals.iloc[z]))
        entr_A.append(entropy.iloc[z][0])
        densidad_A.append(conect.iloc[z][0])
        LE_A.append(abs(LE.iloc[z][0]))
        #lclus_dist_A.append(lclus_distr.iloc[z][0])
        
        
    avals_dist_A = np.array(avals_dist_A)
    grado_A = np.array(grado_A)
    dist_A = np.array(dist_A)
    avals_dist_A = avals_dist_A.reshape((1,avals_dist_A.size))
    grado_A = grado_A.reshape((1,grado_A.size))
    loc_clus_A = np.array(lclus_dist_A)
    loc_clus_A = loc_clus_A.reshape((1,loc_clus_A.size))
    
    for k,ii in enumerate(range(Pts_idx[1][0], Pts_idx[1][1])):
        kk = z + k + 1
        print(ii)
        avals_dist_C.append(avals.iloc[kk])  
        avals_C.append(avals.iloc[kk])
        dist_C.append(dist.iloc[kk])
        idx_2 = np.where(np.array(avals_C[k]) > 0.00001)[0]
        if idx_2.any():
            x = avals_C[k][idx_2[0]]   
        else:
            x = np.nan
        aval_nonulo_C.append(x)
        grado_C.append(grado.iloc[kk])
        lclus_dist_C.append(lclus_distr.iloc[kk])
        Traza_C.append(np.sum(avals.iloc[kk])) 
        Traza_C_std.append(np.std(avals.iloc[kk]))
        entr_C.append(entropy.iloc[kk][0])
        densidad_C.append(conect.iloc[kk][0])        
        LE_C.append(abs(LE.iloc[kk][0]))
        #lclus_dist_C.append(lclus_distr.iloc[kk][0])

    avals_dist_C = np.array(avals_dist_C)
    grado_C = np.array(grado_C)
    dist_C = np.array(dist_C)
    avals_dist_C = avals_dist_C.reshape((1,avals_dist_C.size))
    grado_C = grado_C.reshape((1,grado_C.size))
    loc_clus_C = np.array(lclus_dist_C)
    loc_clus_C = loc_clus_C.reshape((1,loc_clus_C.size))
    
    for j,ii in enumerate(range(Pts_idx[2][0], Pts_idx[2][1])):
        print(ii)
        jj = j + kk + 1
        avals_dist_F.append(avals.iloc[jj])  
        avals_F.append(avals.iloc[jj])
        dist_F.append(dist.iloc[jj])
        idx_2 = np.where(np.array(avals_F[j]) > 0.00001)[0]
        if idx_2.any():
            x = avals_F[j][idx_2[0]]   
        else:
            x = np.nan
        aval_nonulo_F.append(x)
        grado_F.append(grado.iloc[jj])
        lclus_dist_F.append(lclus_distr.iloc[jj])
        Traza_F.append(np.sum(avals.iloc[jj])) 
        Traza_F_std.append(np.std(avals.iloc[jj]))
        entr_F.append(entropy.iloc[jj][0])
        densidad_F.append(conect.iloc[jj][0])
        LE_F.append(abs(LE.iloc[jj][0]))
        #lclus_dist_F.append(lclus_distr.iloc[jj][0])

            
    avals_dist_F = np.array(avals_dist_F)
    avals_dist_F = avals_dist_F.reshape((1,avals_dist_F.size))
    dist_F = np.array(dist_F)
    grado_F = np.array(grado_F)
    grado_F = grado_F.reshape((1,grado_F.size))
    loc_clus_F = np.array(lclus_dist_F)
    loc_clus_F = loc_clus_F.reshape((1,loc_clus_F.size))

    hist_A = np.histogram(avals_dist_A, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    hist_C = np.histogram(avals_dist_C, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    hist_F = np.histogram(avals_dist_F, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    
    hist_grado_A = np.histogram(grado_A, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    hist_grado_C = np.histogram(grado_C, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    hist_grado_F = np.histogram(grado_F, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    
    hist_lclus_A = np.histogram(loc_clus_A, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    hist_lclus_C = np.histogram(loc_clus_C, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    hist_lclus_F = np.histogram(loc_clus_F, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    
    
    # Histograma de autovalores
    chs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    
    xticklabels = [canales_str[j] for j in chs]
    
    fig, axs = plt.subplots(3,1, figsize = (12,6))
    axs[0].stairs(hist_A[0], hist_A[1], color='k')
    axs[0].set_title('$\Lambda$ Histogram A')
    axs[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[0].set_xticklabels(xticklabels)
    axs[1].stairs(hist_C[0], hist_C[1], color='k')
    axs[1].set_title('$\Lambda$ Histogram C')
    axs[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[1].set_xticklabels(xticklabels)
    axs[2].stairs(hist_F[0], hist_F[1], color='k')
    axs[2].set_title('$\Lambda$ Histogram F')
    plt.subplots_adjust(hspace=1)  # Cambia el valor para más o menos espacio
    axs[2].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[2].set_xticklabels(xticklabels)
    plt.savefig(nombre_avals[:-3] + '.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades

    plt.show()
    
   
    # Histograma de grado
    fig, axs = plt.subplots(3,1, figsize = (12,6))
    axs[0].stairs(hist_grado_A[0], hist_grado_A[1], color='k')
    axs[0].set_title('Degree Histogram A')
    axs[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[0].set_xticklabels(xticklabels)
    axs[1].stairs(hist_grado_C[0], hist_grado_C[1], color='k')
    axs[1].set_title('Degre$ Histogram C')
    axs[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[1].set_xticklabels(xticklabels)
    axs[2].stairs(hist_grado_F[0], hist_grado_F[1], color='k')
    axs[2].set_title('Degree Histogram F')
    plt.subplots_adjust(hspace=1)  # Cambia el valor para más o menos espacio
    axs[2].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[2].set_xticklabels(xticklabels)
    plt.savefig(nombre_grado[:-3] + 'png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()


    # Histograma de cluster local
    fig, axs = plt.subplots(3,1, figsize = (12,6))
    axs[0].stairs(hist_lclus_A[0], hist_lclus_A[1], color='k')
    axs[0].set_title('Local cluster Histogram A')
    axs[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[0].set_xticklabels(xticklabels)
    axs[1].stairs(hist_lclus_C[0], hist_lclus_C[1], color='k')
    axs[1].set_title('Local cluster Histogram C')
    axs[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[1].set_xticklabels(xticklabels)
    axs[2].stairs(hist_lclus_F[0], hist_lclus_F[1], color='k')
    axs[2].set_title('Local cluster Histogram F')
    plt.subplots_adjust(hspace=1)  # Cambia el valor para más o menos espacio
    axs[2].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    axs[2].set_xticklabels(xticklabels)
    plt.savefig(nombre_avals[:-3] + '.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades

    plt.show()
    
    
    # Boxplot de Trazas y Std y Var
    fig, axs = plt.subplots(2,1, figsize = (8,4))
    axs[0].boxplot([Traza_A, Traza_C, Traza_F])
    axs[0].set_xticks([1,2,3])
    axs[0].set_xticklabels(['A','C','F'])
    axs[0].set_title('$\Lambda$ Trace')
    axs[1].boxplot([Traza_A_std, Traza_C_std, Traza_F_std])
    axs[1].set_xticks([1,2,3])
    axs[1].set_xticklabels(['A','C','F'])
    axs[1].set_title('$\Lambda$ Std Dev')
    plt.savefig(nombre_avals[:-4] + '_traza.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
    
    # Boxplot de Entropia 
    plt.figure(figsize=(8,4))
    plt.boxplot([entr_A, entr_C, entr_F])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Graph Entropy')
    plt.savefig(nombre_entr[:-3] + 'png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
    
    # Boxplot de Densidad 
    plt.figure(figsize=(8,4))
    plt.boxplot([densidad_A, densidad_C, densidad_F])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Graph Density')
    plt.savefig(nombre_conect[:-3] + 'png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
     
    # Boxplot de Energía del Laplaciano 
    plt.figure(figsize=(8,4))
    plt.boxplot([LE_A, LE_C, LE_F])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Laplacian Energy')
    plt.savefig(nombre_LE[:-3] + 'png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
    
    # Boxplot de segundo aval 
    avals_A = np.array(avals_A)
    avals_C = np.array(avals_C)
    avals_F = np.array(avals_F)

    plt.figure(figsize=(8,4))
    plt.boxplot([avals_A[:,1], avals_C[:,1], avals_F[:,1]])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Segundo aval')
    plt.savefig(nombre_avals[:-4] + '_2do.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
   
    
    plt.figure(figsize=(8,4))
    plt.boxplot([aval_nonulo_A[:], aval_nonulo_C[:], aval_nonulo_F[:]])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Segundo aval nonulo')
    plt.savefig(nombre_avals[:-4] + '_2do_nonulo.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
    
    """
    plt.figure(figsize=(8,4))
    plt.boxplot([lclus_dist_A[:], lclus_dist_C[:], lclus_dist_F[:]])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Local clustering')
    plt.savefig(nombre_lclus[:-4] + '.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
    
    
    plt.figure(figsize=(8,4))
    dist_A = np.where(dist_A > 10000, np.nan, dist_A)
    dist_C = np.where(dist_C > 10000, np.nan, dist_C)
    dist_F = np.where(dist_F > 10000, np.nan, dist_F)
    AA = np.nanmean(dist_A[:], axis = 1)
    CC = np.nanmean(dist_C[:], axis = 1)
    FF = np.nanmean(dist_F[:], axis = 1)
    AA_idx = np.isfinite(AA)
    CC_idx = np.isfinite(CC)
    FF_idx = np.isfinite(FF)
    AA = AA[AA_idx]
    CC = CC[CC_idx]
    FF = FF[FF_idx]
    plt.boxplot([AA, CC, FF])
    plt.xticks([1,2,3])
    #plt.xticklabels(['A','C','F'])
    plt.title('Distancias')
    plt.savefig(nombre_dist[:-4] + '_promedio.png', dpi=300, bbox_inches='tight')  # Ajusta el nombre y el formato según tus necesidades
    plt.show()
    """
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
