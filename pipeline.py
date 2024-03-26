import warnings
# warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import torch
from model import Model
import numpy as np
import pickle
from utils import *
from sklearn.model_selection import train_test_split
################

dataset_name = "Abortiondebate"

with open(f"data/processed/{dataset_name}/textembs.pkl", "rb") as f:
    textembs = pickle.load(f)

X = torch.tensor(np.array([t.numpy().squeeze() for t in textembs]))

num_nodes = X.shape[0]

# random example
# incidence_matrix = (torch.rand((num_nodes, 41)) > 0.5).type(torch.float32)
# edge_index = incidence_matrix.nonzero().t().contiguous()

# matrix ordered by timestamp (e.g. 1,2,3,4,5 1 comes before 2, 2 comes before 3 etc )
with open(f"data/processed/{dataset_name}/matrix.pkl", "rb") as f:
    incidence_matrix = pickle.load(f)

labels = pd.read_csv(f"data/processed/{dataset_name}/id_map.csv")
labels = torch.tensor(labels['label_id'].values) + 1
y = torch.eye(7)[labels]

# train/test dataset - 80-20 ordered by timestamp without shuffling
X_training, X_test, im_training, im_test, y_training, y_test = train_test_split(X, incidence_matrix, y, train_size=0.8, shuffle=False)

# check if we have he with 0 nodes
print(f"{(im_training.T.sum(axis=1) == 0).sum()} empty edges in training set")
print(f"{(im_test.T.sum(axis=1) == 0).sum()} empty edges in test set")
# Remove empty edges from both training e test set
im_training = torch.tensor(im_training.T[(im_training.T.sum(axis=1) > 0)].T)
im_test = torch.tensor(im_test.T[(im_test.T.sum(axis=1) > 0)].T)

print(im_training.shape, im_test.shape)

training_edge_index = im_training.nonzero().T.contiguous()
test_edge_index = im_test.nonzero().T.contiguous()

model = Model(X.shape[1], y.shape[1])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X_training, training_edge_index)
    loss = criterion(y_pred, y_training.argmax(dim=1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Loss: {loss.item()}')
################
