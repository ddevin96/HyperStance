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
import time
import io

from clearml import Task, Logger
import random
################

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def main():
    task = Task.init(project_name="kitemmuorto", task_name="kitemmuorto", reuse_last_task_id=False)

    dataset_name = "sample4000"

    # with open(f"data/processed/{dataset_name}/textembs.pkl", "rb") as f:
        # textembs = pickle.load(f)
    files = os.listdir("/home/ddevin/test/data/embs")
    files.sort()
    textembs = []
    now = time.time()
    for file in files:
        if file.endswith(".pkl"):
            with open(f"/home/ddevin/test/data/embs/{file}", "rb") as f:
                print(f"Loading {file} - {time.time() - now} seconds", flush=True)
                # texts = pickle.load(f)
                texts = CPU_Unpickler(f).load()
                textembs.extend(texts)

    X = torch.tensor(np.array([t.cpu().numpy().squeeze() for t in textembs]))

    num_nodes = X.shape[0]

    # already sorted by timestamp
    # with open(f"data/processed/{dataset_name}/matrix.pkl", "rb") as f:
    now = time.time()
    with open(f"/home/ddevin/test/data/complete/matrix.pkl", "rb") as f:
        incidence_matrix = pickle.load(f)
    print(f"Loaded incidence matrix - {time.time() - now} seconds", flush=True)

    # labels = pd.read_csv(f"data/processed/{dataset_name}/id_map.csv")
    labels = pd.read_csv(f"/home/ddevin/test/data/complete/id_map.csv")
    labels = torch.tensor(labels['label_id'].values)

    y = torch.eye(3)[labels]
    
    # train/test dataset - 80-20 ordered by timestamp without shuffling
    # X_training, X_test, im_training, im_test, y_training, y_test = train_test_split(X, incidence_matrix, y, train_size=0.8, shuffle=False)
    train_size = 0.8
    training_mask = np.zeros(num_nodes)
    training_mask[:int(num_nodes*train_size)] = 1
    training_mask = training_mask.astype(bool)
    test_mask = ~training_mask
    X_training, X_test = X[training_mask], X[~training_mask]
    im_training, im_test = incidence_matrix[training_mask], incidence_matrix
    y_training, y_test = y[training_mask], y[~training_mask]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=10e-6)
    epochs = 1500
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_training, training_edge_index)
        loss = criterion(y_pred, y_training.argmax(dim=1))
        loss.backward()
        optimizer.step()
        Logger.current_logger().report_scalar("Loss", "Training Loss", value=loss.item(), iteration=epoch)
        if epoch % 250 == 0:
            with torch.inference_mode():
                y_pred = model(X, test_edge_index)
                y_pred = y_pred[test_mask]
                test_loss = criterion(y_pred, y_test.argmax(dim=1))
                y_pred = y_pred.argmax(dim=1)
                y_true = y_test.argmax(dim=1)
                acc = accuracy_score(y_true, y_pred) * 100
                f1 = f1_score(y_true, y_pred, average='weighted') * 100
                print(f'Epoch {epoch}: Training Loss: {loss.item()} - Test Loss: {test_loss.item():.2f} - Accuracy {acc:.2f} - F1 {f1:.2f}', flush=True)
                Logger.current_logger().report_scalar("Loss", "Test Loss", value=test_loss.item(), iteration=epoch)
                Logger.current_logger().report_scalar("Metrics", "Accuracy", value=acc, iteration=epoch)
                Logger.current_logger().report_scalar("Metrics", "F1", value=f1, iteration=epoch)
                model.eval()

    ################

if __name__ == "__main__":
    main()
