#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:55:42 2024

@author: mariapau
"""

import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
import mne



############################################################################
def GSA(g, A, L):
    aval,avec = np.linalg.eigh(L)
    #aval = np.sort(aval)
    avalA,avecA = np.linalg.eigh(A)
    idx = np.argmax(abs(avalA))
    ec = avecA[:,idx]
   
    """
    aval_A,avec_A = np.linalg.eig(A)
    idx = np.argmax(abs(aval_A))
    aval = np.sort(aval)
    ec_P = avec_A[:,idx]   # puede haber mas de uno máximos, ahi hay que chequear por el asociado al avec de entradas positivas
    try:
        ec = gtc.centrality.eigenvector(g, g.ep["peso"] )
    except:
        ec = (np.nan, [np.nan]*19)
    """
    Spec_ratio = avalA[idx]
    Spec_gap = aval[-1]-aval[-2]
    grado_distr = np.diagonal(L)
    idx = np.where(aval > 0.0000001)[0]
    if idx.any():
        idx = idx[0]
        AC = aval[idx]
    else:
        AC = np.nan
    
    # Energía del laplaciano
    m = 0
    n = 0
    LE = 0
    for v in g.vertices():
      n = n + 1
    for e in g.edges():
      m = m + 1
    for i in range(aval.size):
      LE += np.sum(abs(aval[i] - (2*m/n)))
  
    return ec, Spec_ratio, Spec_gap, LE, grado_distr, AC 
  



############################################################################

def armo_laplaciano(cov, threshold):
   
   # Genero Laplaciano     
   cov_umb = cov.copy()
   cov_umb = np.abs(cov_umb)
   idx0 = np.where(cov < threshold)
   cov_umb[idx0[0], idx0[1]] = 0                # Matriz de covarianza umbralizada idx = np.where(mat_cov_d0 < threshold)
   #idx1 = np.where(cov > threshold)
   #cov_umb[idx1[0], idx1[1]] = 1                # Matriz de covarianza binarizada
   d_0 = np.diag(cov_umb)
   Adj = cov_umb-np.diag(d_0)                   # Matriz de Adyacencia
   
   diagonal = np.array([sum(Adj[i,:] for i in range(Adj.shape[0]))])[0]  # Genero matriz de grado sobre matriz umbralizada
   D = np.diag(diagonal)                        # Genero matriz de grado
   L = D - Adj                                  # Laplaciano
   #L = np.abs(L)
   #L[L < 0] = 0                  
   return Adj, L
      

############################################################################

def genero_grafo(cov, threshold, canales, canales_str, ploteo = 1):
  
    A, L = armo_laplaciano(cov, threshold)

    # Inicializar grafo no dirigido
    g = gt.Graph(directed=False)
    # Añadir vértices al grafo
    vertex_map = {node: g.add_vertex() for node in canales_str}      
    #g.vertex_properties["name"] = g.new_vertex_property("string")
  
    # Crear una propiedad para las etiquetas de los nodos
    vertex_labels = g.new_vertex_property("string")
    for node, vertex in vertex_map.items():
         vertex_labels[vertex] = node  # Asignar el nombre del canal al vértice
  
    # Crear un mapa de propiedades para almacenar los pesos de las aristas
    peso_arista = g.new_edge_property("float")    
    # Encontrar las posiciones donde la matriz supera el umbral
    idx = np.where(A > 0)    
    # Crear la lista de aristas, excluyendo bucles (autoenlaces)
    for i, j in zip(idx[0], idx[1]):
        if i != j:
            # Agregar la arista al grafo
            e = g.add_edge(i, j)
            # Asignar el peso de la arista
            peso_arista[e] = A[i, j]    
    # Asignar el mapa de propiedades de peso al grafo
    g.edge_properties["peso"] = peso_arista    
    # Remover aristas paralelas
    gt.remove_parallel_edges(g)
    
    # Define las regiones corticales y los colores
    region_colors = {
        "frontales": "#FFC0CB",  # Rosa
        "centrales": "#FFFF00",  # Amarillo
        "temporales": "#87CEFA",  # Azul
        "parietales": "#32CD32",  # Verde
        "occipitales": "#9370DB"  # Morado
    }
    
    # Define los canales en cada región cortical (según el sistema 10-20)
    regiones_corticales = {
        "frontales": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
        "centrales": ["C3", "Cz", "C4"],
        "temporales": ["T3", "T4", "T5", "T6"],
        "parietales": ["P3", "Pz", "P4"],
        "occipitales": ["O1", "Oz", "O2"]
    }
    
    # Crear una propiedad para los colores de los nodos
    vertex_colors = g.new_vertex_property("string")
    
    # Asignar colores según la región cortical
    for node, vertex in vertex_map.items():
        color_asignado = "#D3D3D3"  # Color por defecto (gris claro)
        for region, canales in regiones_corticales.items():
            if node in canales:
                color_asignado = region_colors[region]
                break
        vertex_colors[vertex] = color_asignado  # Asignar color al nodo


    # Cargo etiquetas de los nodos
    if ploteo == True:
       
      # Normalizar los pesos para utilizarlos como anchos de aristas
      # Puedes ajustar el factor de escala (por ejemplo, multiplicando por 5) para hacer más visibles los anchos
      max_peso = max(peso_arista.a)
      ancho_arista = g.new_edge_property("float")
      ancho_arista.a = 5 * (peso_arista.a / max_peso)  # Ajusta el factor de escala a tu gusto

      # Cargar el montaje del sistema 10-20
      montage = mne.channels.make_standard_montage('standard_1020')
      pos = montage.get_positions()['ch_pos']  # Diccionario de posiciones de los electrodos
      

      # Crear una propiedad para las posiciones de los vértices
      vertex_positions = g.new_vertex_property("vector<double>")
      for node, vertex in vertex_map.items():
          vertex_positions[vertex] = pos[node][:2]  # Coordenadas (x, y)


      # Dibuja el grafo, asignando los anchos y colores de aristas
      gt.graph_draw(
          g,
          pos=vertex_positions,
          vertex_text=vertex_labels,  # Usar los nombres de los canales como texto de los nodos
          vertex_fill_color=vertex_colors,
          vertex_size=40,
          edge_pen_width=2,
          output_size=(800, 600),
          #output="grafo_eeg.png",  # También guarda la imagen en un archivo
          )

    return g, A, L

#############################################################################

def calculo_grafo(g, pares):

  print('========================================')
  #print('Numero de nodos: {:d}'.format(g.num_vertices()))
  #print('Numero de enlaces: {:d}'.format(g.num_edges()))
 
  densidad = 2*g.num_edges()/float(g.num_vertices()*(g.num_vertices()-1))
  #print('Conectancia: {:.3f}'.format(conectancia_0))

  clustering_global = gt.global_clustering(g)[0]
  clustering_local = gt.local_clustering(g)
  #lclus = gt.vertex_average(g, clustering_local)
    
  return densidad, clustering_global, clustering_local.a

############################################################################

def calc_distancias(g, pares):
    d = np.zeros((len(pares),))

    # Calcular las distancias más cortas desde un vértice de origen a todos los demás
    for n,k in enumerate(pares):
        d[n] = gt.shortest_distance(g, source=g.vertex_index[k[0]], target=g.vertex_index[k[1]])
    # Imprimir las distancias desde v1 a todos los demás vértices 
    
    return d
    
############################################################################
def calculo_entropia(g,grado):
    # Obtiene los grados de los nodos
    
    # Cuenta la frecuencia de cada grado
    grado_counts = np.bincount(grado.astype(int))
    
    # Calcula la distribución de grados
    p_k = grado_counts / sum(grado_counts)
    
    # Calcula la entropía
    entropia = -np.sum([p * np.log(p) for p in p_k if p > 0])
    
    return entropia