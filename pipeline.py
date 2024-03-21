import torch
from model import Model
import numpy as np
import pickle
from utils import *
################
with open("data/processed/AntiVegan/textembs.pkl", "rb") as f:
    textembs = pickle.load(f)

X_n = torch.tensor(np.array([t.numpy().squeeze() for t in textembs]))

num_nodes = X_n.shape[0]

# random example
# incidence_matrix = (torch.rand((num_nodes, 41)) > 0.5).type(torch.float32)
# edge_index = incidence_matrix.nonzero().t().contiguous()

with open("data/processed/AntiVegan/matrix.pkl", "rb") as f:
    incidence_matrix = pickle.load(f)

incidence_matrix = torch.tensor(incidence_matrix)
edge_index = incidence_matrix.nonzero().t().contiguous()
print(X_n.shape)
print(edge_index)

model = Model(X_n.shape[1], 6)

print(model(X_n, edge_index))
################