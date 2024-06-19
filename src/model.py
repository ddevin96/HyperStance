import torch
from torch import nn
from torch_geometric.nn import HypergraphConv
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.hypergraph_conv_1 = HypergraphConv(in_channels, in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, X, edge_index):
        y = self.dropout(X)
        y = self.hypergraph_conv_1(y, edge_index)
        y = self.linear(y)
        y = nn.functional.softmax(y, dim=1)
        return y

class Model2(nn.Module):
    def __init__(self, embedding_dim, network_feature_dim, hidden_dim, output_dim):
        super(Model2, self).__init__()
        self.text_fc = nn.Linear(embedding_dim, hidden_dim)
        self.network_fc = nn.Linear(network_feature_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text_embeddings, network_structure):
        text_out = F.relu(self.text_fc(text_embeddings))
        network_out = F.relu(self.network_fc(network_structure))
        combined = torch.cat((text_out, network_out), dim=1)
        output = self.fc(combined)
        return output


class HypergraphTextPretrainModel(nn.Module):
    def __init__(self, embedding_dim, hyperedge_dim):
        super(HypergraphTextPretrainModel, self).__init__()
        self.hypergraph_conv = HypergraphConv(in_channels=embedding_dim, out_channels=embedding_dim)
        self.text_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, node_features, incidence_matrix):
        hypergraph_out = self.hypergraph_conv(node_features, incidence_matrix)
        text_out = self.text_fc(node_features)
        combined = torch.cat([hypergraph_out, text_out], dim=1)
        out = self.fusion_fc(combined)
        return out

# Define a new model class for fine-tuning
class FineTunedHypergraphModel(nn.Module):
    def __init__(self, pretrain_model, num_classes):
        super(FineTunedHypergraphModel, self).__init__()
        self.pretrain_model = pretrain_model
        # Freeze the layers of the pre-trained model
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.pretrain_model.fusion_fc[-1].out_features, num_classes)

    def forward(self, node_features, incidence_matrix):
        features = self.pretrain_model(node_features, incidence_matrix)
        out = self.classifier(features)
        return out