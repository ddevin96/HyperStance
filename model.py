import torch
from torch import nn
from torch_geometric.nn import HypergraphConv

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        self.hypergraph_conv_1 = HypergraphConv(in_channels, out_channels)
    
    def forward(self, X, edge_index):
        y = self.hypergraph_conv_1(X, edge_index)
        y = nn.functional.softmax(y, dim=1)
        return y
    