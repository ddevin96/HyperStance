import argparse
import os
import numpy as np
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from model import Model, Model2, HypergraphTextPretrainModel, FineTunedHypergraphModel
from clearml import Task, Logger
import pandas as pd

# Self-supervised loss and optimizer
def self_supervised_loss(reconstructed, original):
    return torch.nn.functional.mse_loss(reconstructed, original)

def evaluate_model(model, node_features, incidence_matrix, labels):
	model.eval()
	all_labels = []
	all_preds = []
	with torch.no_grad():
		outputs = model(node_features, incidence_matrix)
		_, preds = torch.max(outputs, 1)

		all_labels.extend(labels.cpu().numpy())
		all_preds.extend(preds.cpu().numpy())

		accuracy = accuracy_score(all_labels, all_preds)
		precision = precision_score(all_labels, all_preds, average='weighted')
		recall = recall_score(all_labels, all_preds, average='weighted')
		f1 = f1_score(all_labels, all_preds, average='weighted')
		conf_matrix = confusion_matrix(all_labels, all_preds)
		class_report = classification_report(all_labels, all_preds)

	return accuracy, precision, recall, f1, conf_matrix, class_report

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Hypergraphs')
    parser.add_argument('-s', '--size', type=int, help='Size of dataset', default=1000)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=1000)
    parser.add_argument('-o', '--optimizer', type=str, help='Optimizer choice', default='adam')
    parser.add_argument('-l', '--loss', type=str, help='Loss choice', default='mse')
    parser.add_argument('-d', '--device', type=str, help='Device', default='cuda')
    parser.add_argument('-m', '--model', type=str, help='Model', default='HypergraphTextPretrainModel')
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=0.001)
    args = parser.parse_args()

    if args.size is not None:
        size = args.size

    if args.epochs is not None:
        epochs = args.epochs

    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    else:
        raise ValueError("Invalid device choice")
    
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    

    task = Task.init(
        project_name="test", 
        task_name="HG train", 
        reuse_last_task_id=False,
        tags = [str(args.size), args.loss, args.optimizer, args.model, str(args.learning_rate)]
        )

    path = f"../data/complete/sample{size}"
    if not os.path.exists(path):
        print("Path does not exist")
        exit()

    embeddings = pickle.load(open(f"{path}/textembs.pkl", "rb"))
    matrix = pickle.load(open(f"{path}/matrix.pkl", "rb"))

    embs = torch.tensor(np.array([t.cpu().numpy().squeeze() for t in embeddings]))
    incidence_matrix = torch.tensor(matrix)
    edge_index = incidence_matrix.nonzero().T.contiguous() # (2,E)

    X = embs
    node_features = embs
    num_nodes = X.shape[0]    

    if args.model == 'model1':
        y = torch.randint(0, 3, (X.shape[0],))
        y = torch.eye(3)[torch.randint(0, 3, (X.shape[0],))]
        model = Model(X.shape[1], y.shape[1])
    elif args.model == 'model2':
        y = torch.randint(0, 3, (X.shape[0],))
        y = torch.eye(3)[torch.randint(0, 3, (X.shape[0],))]
        model = Model2(X.shape[1], y.shape[1])

    elif args.model == 'HypergraphTextPretrainModel':
        # pretrain_model
        print(f"node_features: {node_features.size(1)}, edge_index: {edge_index.size(1)}")
        model = HypergraphTextPretrainModel(embedding_dim=node_features.size(1), hyperedge_dim=edge_index.size(1))
    else:
        raise ValueError("Invalid model choice")
    

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=10e-6)
    else:
        raise ValueError("Invalid optimizer choice")

    model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)

    # Pretraining loop
    for epoch in range(0, epochs+1):
        model.train()
        optimizer.zero_grad()
        output = model(node_features, edge_index)
        # loss = self_supervised_loss(output, node_features)
    
        if args.loss == 'mse':
            loss = torch.nn.functional.mse_loss(output, node_features)
        elif args.loss == 'mae':
            loss = torch.nn.functional.l1_loss(output, node_features)
        else:
            raise ValueError("Invalid loss choice")

        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            # print(f"Pretrain Epoch {epoch+1}, Loss: {loss.item()}")
            Logger.current_logger().report_scalar("Loss", "Pretrain Loss", value=loss.item(), iteration=epoch)
            
    # save the model
    torch.save(model.state_dict(), f"../data/checkpoint/pretrain_model{size}.pth")

############################################################################################################

    # Fine tune the model with labeled data
    annotations = ['favor', 'against', 'unknown']
    csv_file = "../data/aggregated/aggregatedBefore1yearFA.csv"
    data = pd.read_csv(f"{csv_file}")
    data = data.head(size)
    data = data[data.author != "AutoModerator"]
    data['label_id'] = data['gemmamlabel'].apply(lambda x: annotations.index(x) if x in annotations else -1)

    aggregated_data = data.groupby('author').agg({
		'created_utc': lambda x: ', '.join(map(str, x)),
		'id': lambda x: ', '.join(x),
		'link_id': lambda x: ', '.join(x),
		'subreddit': lambda x: ', '.join(x),
		'parent_id': lambda x: ', '.join(x),
		'content': lambda x: ', '.join(x),
		'gemmamlabel': lambda x: ', '.join(x),
		'label_id': lambda x: x.value_counts().idxmax() # i care about this
	}).reset_index()


    labels = torch.tensor(aggregated_data['label_id'].values).to('cuda')
    # labels.to(device)

    num_classes = len(torch.unique(labels))

    fine_tune_model = FineTunedHypergraphModel(model, num_classes)
    fine_tune_model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=0.001)

    # print(f"node_features: {node_features.size()}, edge_index: {edge_index.size()}, labels: {labels.size()}")
    # print(f"num_classes: {num_classes}")
    # print(f"fine_tune_model: {fine_tune_model}")
    # print(f"optimizer: {optimizer}")

    # Fine-tuning loop
    num_fine_tune_epochs = 10000
    for epoch in range(num_fine_tune_epochs):
        fine_tune_model.train()
        # print(f"fine model: {fine_tune_model.is_cuda}")
        optimizer.zero_grad()
        output = fine_tune_model(node_features, edge_index)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Fine-tune Epoch {epoch+1}, Loss: {loss.item()}")
            Logger.current_logger().report_scalar("Loss", "Fine-tune Loss", value=loss.item(), iteration=epoch)

            #### METRICS
            accuracy, precision, recall, f1, conf_matrix, class_report = evaluate_model(fine_tuned_model, node_features, incidence_matrix, labels)
            print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
            print(f'Confusion Matrix:\n{conf_matrix}')
            print(f'Classification Report:\n{class_report}')

    # FINAL METRICS
    accuracy, precision, recall, f1, conf_matrix, class_report = evaluate_model(fine_tuned_model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')
