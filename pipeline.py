# import warnings
# warnings.filterwarnings("ignore")
import torch
from model import Model
import numpy as np
import pickle
from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

################

def main():
    dataset_name = "Abortiondebate"

    with open(f"data/processed/{dataset_name}/textembs.pkl", "rb") as f:
        textembs = pickle.load(f)

    X = torch.tensor(np.array([t.numpy().squeeze() for t in textembs]))

    num_nodes = X.shape[0]

    # already sorted by timestamp
    with open(f"data/processed/{dataset_name}/matrix.pkl", "rb") as f:
        incidence_matrix = pickle.load(f)

    labels = pd.read_csv(f"data/processed/{dataset_name}/id_map.csv")
    labels = torch.tensor(labels['label_id'].values) + 1
    labels = np.random.randint(0, 6, len(labels)) # Mock labels

    # y = torch.eye(7)[labels] # Real y
    y = torch.eye(6)[labels] # Mock y

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=10e-6)
    epochs = 3000
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_training, training_edge_index)
        loss = criterion(y_pred, y_training.argmax(dim=1))
        loss.backward()
        optimizer.step()
        if epoch % 250 == 0:
            with torch.inference_mode():
                y_pred = model(X_test, test_edge_index)
                test_loss = criterion(y_pred, y_test.argmax(dim=1))
                y_pred = y_pred.argmax(dim=1)
                y_true = y_test.argmax(dim=1)
                acc = accuracy_score(y_true, y_pred) * 100
                f1 = f1_score(y_true, y_pred, average='weighted') * 100
                print(f'Epoch {epoch}: Training Loss: {loss.item()} - Test Loss: {test_loss.item():.2f} - Accuracy {acc:.2f} - F1 {f1:.2f}')
                model.eval()

    ################

if __name__ == "__main__":
    main()
