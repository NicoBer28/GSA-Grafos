#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:20:44 2025

@author: mariapau
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import graph_tool.all as gt
import numpy as np

# Función para convertir un grafo de graph-tool a un objeto Data de PyTorch Geometric
def graph_tool_to_pyg(graph, labels):
    edge_index = torch.tensor([[int(e.source()), int(e.target())] for e in graph.edges()], dtype=torch.long).t().contiguous()
    x = torch.tensor(graph.vp['features'].get_2d_array(range(graph.num_vertices())).T, dtype=torch.float)
    y = torch.tensor([labels[graph] if graph in labels else 0], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# Definir la arquitectura de la GNN
class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Obtener features de un grafo
def feature_graphs(num_graphs=100, num_nodes=20, num_features=5):
    graphs = []
    g.vp['features'] = g.new_vertex_property('vector<float>')
    
    for v in g.vertices():
        # Obtener los pesos de todas las aristas conectadas al nodo
        edge_weights = [g.ep['weight'][e] for e in v.all_edges()]
        
        if edge_weights:  # Si el nodo tiene conexiones
            total_weight = sum(edge_weights)  # Suma de pesos
            avg_weight = np.mean(edge_weights)  # Promedio de pesos
            max_weight = max(edge_weights)  # Máximo peso
            min_weight = min(edge_weights)  # Mínimo peso
            std_weight = np.std(edge_weights)  # Desviación estándar
        else:  # Si el nodo no tiene conexiones, usa valores neutros
            total_weight, avg_weight, max_weight, min_weight, std_weight = 0, 0, 0, 0, 0
        
    # Asignar el vector de características al nodo
    g.vp['features'][v] = [total_weight, avg_weight, max_weight, min_weight, std_weight]

    graphs.append(graph_tool_to_pyg(g, labels))
    return graphs

# Cargar datos
graphs = generate_graphs()
dataloader = DataLoader(graphs, batch_size=16, shuffle=True)

# Definir modelo y entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNClassifier(input_dim=5, hidden_dim=16, output_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(20):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

print("Entrenamiento completado")
