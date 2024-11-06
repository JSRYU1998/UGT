# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 03:54:45 2023

@author: RYU
"""

import torch
import torch_geometric as pyg

#Define graph dataset for test
class GraphDataset_Darcy(pyg.data.Dataset):
    def __init__(self, position_vectors, node_feature_vectors, target_vectors, K):
        #position_vectors.shape : (num_data, s**2, 2)
        #node_feature_vectors.shape : (num_data, s**2, 1)
        #target_vectors.shape : (num_data, s**2, 1)
        super().__init__()
        self.K = K
        
        self.positions = position_vectors #torch_tensor
        self.node_features = node_feature_vectors #torch_tensor
        self.targets = target_vectors #torch_tensor
        
    def len(self):
        return len(self.positions)
    
    def get(self, idx):
        node_loc = self.positions[idx]
        node_ft = self.node_features[idx]
        
        # construct the graph using the KNN
        edge_index = pyg.nn.knn_graph(x=node_loc, k=self.K, loop=False) #True
        
        # return the graph with features
        graph = pyg.data.Data(
            x = torch.concat([node_loc, node_ft], dim=1), #original : node_ft
            edge_index=edge_index,
            pos=node_loc
        )
        if self.targets is not None:
            graph.targets = self.targets[idx]
        return graph