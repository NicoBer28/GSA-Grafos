import numpy as np
import graph_tool.all as gt
import mne


def DGSA(L):
    aval, avec = np.linalg.eigh(L)

    idx = np.where(aval > 0.0000001)[0]
    if idx.any():
        AC = aval[idx[0]]
    else:
        AC = np.nan

    Spec_gap = aval[-1] - aval[-2]

    if len(idx) > 0:
        autovector_principal = avec[:, idx[0]]
    else:
        autovector_principal = avec[:, 0]

    #importancia estructural del electrodo (reemplaza ec)
    ec_magnitud = np.abs(autovector_principal)

    #va de -pi a pi. Si es >0 envia informacion, si es <0 recibe informacion
    ec_fase = np.angle(autovector_principal)

    return AC, Spec_gap, ec_magnitud, ec_fase 


def calculo_grafo_dirigido(g, A):
    
    V = g.num_vertices()
    E = g.num_edges()
    

    # El máximo de conexiones posibles si es dirigido es V * (V - 1)
    if V > 1:
        densidad = E / float(V * (V - 1))
    else:
        densidad = 0.0


    clustering_global = gt.global_clustering(g)[0]
    clustering_local = gt.local_clustering(g).a
    
    # representa de 0 a 1 la proporción de aristas que son bidireccionales en el grafo dirigido.
    reciprocidad = gt.edge_reciprocity(g)
    
    in_degree = np.sum(A, axis=0)
    out_degree = np.sum(A, axis=1)
    flujo = out_degree - in_degree
    
    # PageRank
    pr = gt.pagerank(g, weight=g.ep["peso"]).a

    return densidad, clustering_global, clustering_local, reciprocidad, in_degree, out_degree, flujo, pr



def armo_laplaciano_dirigido(cov,threshold, q):
   A = np.copy(cov)
   A[A < 0] = 0.0
   A[np.abs(A) < threshold] = 0.0
   
   #parte simetrica
   A_sym = (A + A.T) / 2.0
   #parte asimetrica
   Theta = A - A.T
   
   #usamos formula de euler
   #queda una matriz compleja, "simetrica" (con los conjugados)
   H = A_sym * np.exp(1j * 2 * np.pi * q * Theta)

   #matriz de grado
   grados = np.sum(A_sym, axis=1)
   D = np.diag(grados)

   L = D - H

   return A, L

############################################################################

def genero_grafo_dirigido(cov, threshold, canales, canales_str, q, ploteo = 1):
  
    A, L = armo_laplaciano_dirigido(cov, threshold, q)

    # Inicializar grafo no dirigido
    g = gt.Graph(directed=True)
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