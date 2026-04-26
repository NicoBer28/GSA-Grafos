#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:01:39 2024

@author: mariapau
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:40:50 2024

@author: mariapau
 """

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import Grafos_Paula_lib as gpl
import scipy as sp
from scipy.stats import kruskal, mannwhitneyu
import itertools
import seaborn as sns


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
def grafico_cajas(df, parametro):
    # Filtrar el DataFrame para el parámetro 'potencia'
    df_param = df[df['Parámetro'] == parametro]
    # Asegurarse de que la columna 'Valores' no tenga listas (aplanar si es necesario)
    df_param = df_param.explode('Valores')  # Convierte listas en filas
    df_param['Valores'] = df_param['Valores'].astype(float)  # Asegúrate de que sean numéricos
    
    # Crear el boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_param, x='Ritmo', y='Valores', hue='Grupo', palette='Set2')
    
    plt.title('$PSD \ by \ Rhythm \ and \ Group$', fontsize=22)
    plt.ylabel('$PSD \ [\mu ^2/Hz]$', fontsize=22)
    plt.xticks(range(5), ['$Delta$', '$Theta$', '$Beta$', '$Alpha$', '$Gamma$'], fontsize=16)
    plt.xlabel('$Rhythms$', fontsize=22)
    plt.legend(title='Group', fontsize=16)
    plt.tick_params(axis='y', labelsize=18) 
    plt.subplots_adjust(bottom=0.12)  # Aumentar margen inferior
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    p_value = []
    for n,ritmo in enumerate(ritmos):
        b1 = df_param[(df_param['Grupo'] == 'A') & (df_param['Ritmo'] == ritmo)]
        b2 = df_param[(df_param['Grupo'] == 'C') & (df_param['Ritmo'] == ritmo)]
        b3 = df_param[(df_param['Grupo'] == 'F') & (df_param['Ritmo'] == ritmo)]
    
        statistic, p = kruskal(b1['Valores'], b2['Valores'], b3['Valores'])
        p_value.append(p)
        print('Ritmo: ', ritmo, 'p_value: ', p)
        
     
        pares = list(itertools.combinations(range(len([b1, b2, b3])), 2))
        
        # Calcular el nuevo alfa usando Bonferroni
        alfa_original = 0.05
        alfa_bonferroni = alfa_original / len(pares)
        
        """    
        print("\nComparaciones de a pares con corrección de Bonferroni (alfa ajustado = {:.5f}):".format(alfa_bonferroni))
        for (i, j) in pares:
                # Prueba de Mann-Whitney U para cada par de grupos
                stat, p = mannwhitneyu(potencia[i], potencia[j])
                print(f"Grupo {i+1} vs Grupo {j+1}: estadístico U = {stat}, p-valor = {p}")
                
                # Interpretar el resultado con el alfa ajustado
                if p < alfa_bonferroni:
                    print(f"  Diferencia significativa entre Grupo {i+1} y Grupo {j+1} (p < {alfa_bonferroni})")
                else:
                    print(f"  No hay diferencia significativa entre Grupo {i+1} y Grupo {j+1} (p >= {alfa_bonferroni})")
       
        """
    
    amp = 0.7
    
    p_value = np.array(p_value)
    plt.plot(range(5), 0.94*amp*(np.where(p_value<0.0167, 1, np.nan)), 'k*')
    plt.show()
    
    # Guardar el topomap como archivo
    output_file = "/Users/mariapau/Conicet/Grafos_IAM /Codigo/Power.pdf"
    plt.savefig(output_file, dpi=300)  # Guardar como PNG con alta resolución
 
    
############################################################################



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#       MAIN                         ###########################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Inicializo estructuras
canales_str = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

pares = (7,11)

os.chdir('./')
print(os.getcwd())
    

ritmos = ['delta', 'theta', 'beta', 'alpha', 'gamma']
ritmos_str = ['$\delta$', '$\theta$', '$\beta$', '$\alpha$', '$\gamma$']
 
# Estructura de datos para agrupar métricas
grupos = {"A": {}, "C": {}, "F": {}}
   

potencia = [[],[],[],[],[]]
EC = [[],[],[],[],[]]
lclus = [[],[],[],[],[]]
deg = [[],[],[],[],[]]
b_mean = []
b_std = []

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Aca potencia es una lista de 5 campos, cada campo es un ritmo
for n,ritmo in enumerate(ritmos):
    nombre_Pot = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/0.9/Power.pkl'
    nombre_EC = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/0.9/EC.pkl'
    nombre_lclus = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/0.9/local_clustering.pkl'
    nombre_deg = '/Users/mariapau/Conicet/Grafos_IAM /' + ritmo + '/0.9/degree_distribution.pkl'
    print(nombre_Pot)
    with open(nombre_Pot, 'rb') as f:
        a = pickle.load(f) # deserialize using load()
        potencia[n].append(np.array(a).flatten())
    with open(nombre_EC, 'rb') as f:
        a = pickle.load(f) # deserialize using load()
        EC[n].append(abs(np.array(a).flatten()))
    with open(nombre_lclus, 'rb') as f:
        a = pickle.load(f) # deserialize using load()
        lclus[n].append(np.array(a).flatten())
    with open(nombre_deg, 'rb') as f:
        a = pickle.load(f) # deserialize using load()
        deg[n].append(np.array(a).flatten())
        

############################################################################        
def plot_stairs(grupos, metrica):
    escalas = {'AC' : 21, 'LE': 90, 'Density': 1.2,  'lclus': 1.1,  'EC': 0.4, 
               'deg':19, 'g_clust': 19, 'EC_prom': 0.8, 'Distance': 4}
    titulos = {'AC' : 'AC', 'LE': 'LE', 'deg': 'Degree \ Centrality', 'Entropy': 'Entropy', 'SR': 'SR', 
               'lclus': 'Local \ Clustering', 'g_clust': 'Glob \ Clust', 'EC': 'Eigenvector \ Centrality', 'EC_prom': 'av \ EC', 'Distance': 'Path \ Length'}
    plt.figure(figsize=(4,10))
    plt.suptitle(f'${titulos[metrica]},$\n $th = 0.9$', fontsize=16)
    equis = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    
    for n, ritmo in enumerate(ritmos):
        b_mean = []  # Reiniciar b_mean en cada iteración de ritmo
        lista_claves = list(grupos.keys())
        for grupo in lista_claves[:-1]: 
            aux = grupos[grupo][metrica][ritmo]  # Obtener los datos correspondientes a ritmo y grupo
            
            # Verifica la forma de los datos
            print(f'grupo: {grupo}, ritmo: {ritmo}, shape: {aux.shape}')
            
            # Asegúrate de que 'aux' sea 2D (con las dimensiones correctas)
            if aux.ndim == 2:
                # Calcular las medias de las diferentes secciones
                b_mean.append([
                    np.mean(aux[:,:7]),
                    np.mean(aux[:,7:10]),
                    np.mean(aux[:,10:14]),
                    np.mean(aux[:,14:17]),
                    np.mean(aux[:,17:])
                ])
            else:
                print(f"Error: 'aux' no tiene las dimensiones esperadas: {aux.ndim}")
        
        # Verifica que b_mean tenga datos antes de intentar graficar
        print(f'Valores de b_mean para ritmo {ritmo}: {b_mean}')
        
        # Crear el gráfico
        plt.subplot(5, 1, n+1)
        plt.ylabel(f'${ritmo}$', fontsize=14)
        plt.stairs(b_mean[0], linewidth=2, label = 'A')  # Usar `b_mean` directamente
        plt.stairs(b_mean[1], linewidth=2, label = 'C')  # Usar `b_mean` directamente
        #plt.stairs(b_mean[2], linewidth=2)  # Usar `b_mean` directamente
        plt.xticks(equis, ['F', 'C', 'T', 'P', 'O'], fontsize=14)
        plt.ylim(0,escalas[metrica])
    plt.tight_layout()  # Ajustar la disposición de los subgráficos
    plt.legend()
    plt.show()
    plt.savefig("/Users/mariapau/Conicet/Grafos_IAM /Codigo/stairs_" + metrica + ".pdf")

############################################################################
# Inicializo el diccionario y cargo los valores #############################
# Estructura de datos para agrupar métricas
grupos = {"A": {}, "C": {}, "F": {}}
plt.close('all')

# Crear un Diccionario a partir de potencia
lista = [('A', [0, 35*4*19]), ('C', [35*4*19, 65*4*19]), ('F', [65*4*19, 87*4*19])]
params = ['potencia', 'EC', 'lclus', 'deg']

for g in lista:
    for param in params[1:]:
        # Asegurar que el subdiccionario para cada parámetro esté inicializado
        if param not in grupos[g[0]]:
            grupos[g[0]][param] = {}
        
        for n, ritmo in enumerate(ritmos):
            # Asegurar que cada ritmo también tenga un subdiccionario
            if ritmo not in grupos[g[0]][param]:
                grupos[g[0]][param][ritmo] = {}

            #Asigno los valores
            if param == 'EC':
                grupos[g[0]][param][ritmo] = np.array(EC[n][0][g[1][0]:g[1][1]]).reshape(int(g[1][1]/19)-int(g[1][0]/19),19)
                print(g[1][0], g[1][1])
            elif param == 'lclus':
                grupos[g[0]][param][ritmo] = np.array(lclus[n][0][g[1][0]:g[1][1]]).reshape(int(g[1][1]/19)-int(g[1][0]/19),19)
            elif param == 'deg':
                grupos[g[0]][param][ritmo] = np.array(deg[n][0][g[1][0]:g[1][1]]).reshape(int(g[1][1]/19)-int(g[1][0]/19),19)

for g in lista:
    #inicializo
    if 'potencia' not in grupos[g[0]]:
        grupos[g[0]]['potencia'] = {}
        #asigno
        for n, ritmo in enumerate(ritmos):
            grupos[g[0]]['potencia'][ritmo] = potencia[n][0][int(g[1][0]/19):int(g[1][1]/19)]
        

        
# Convertir la lista de diccionarios a DataFrame
# Lista para almacenar las filas del DataFrame
filas = []

# Recorrer la estructura del diccionario
for grupo, parametros in grupos.items():
    for param, ritmos in parametros.items():
        for ritmo, valores in ritmos.items():
            # Crear una fila con el grupo, parámetro, ritmo y valores
            filas.append({
                "Grupo": grupo,
                "Parámetro": param,
                "Ritmo": ritmo,
                "Valores": valores
            })

# Crear el DataFrame
df = pd.DataFrame(filas)

# Mostrar el DataFrame
print(df)

grafico_cajas(df,'potencia')

plot_stairs(grupos, 'deg')

