import os 
import pandas as pd
import pickle
import torch
from transformers import BertModel, BertTokenizer
import hypernetx as hnx
import numpy as np
import argparse

def hgToIncidenceMatrix(inputFile, outputFile):
    l = []
    with open(inputFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l.append(set(map(int, line.split(','))))
    H = hnx.Hypergraph(l)
    print(H.shape)
    with open(outputFile, 'wb') as f:
        pickle.dump(H.incidence_matrix().toarray(), f, pickle.HIGHEST_PROTOCOL)

def save_embs_old(df, out):
    model_name = 'bert-base-uncased'  # You can use other pre-trained BERT models as well
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = df[["content"]]
    text_embs = []
    pickles = 0
    for index, row in df.iterrows():
        t = str(row["content"])
        t = t.lower()
        inputs = tokenizer(t, truncation=True, max_length=512, return_tensors="pt").to(device)
        # Get the outputs from the BERT model
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the embeddings (output of the [CLS] token)
        embeddings = outputs.last_hidden_state[:, 0, :]
        text_embs.append(embeddings)
    pickle.dump(text_embs, open(f"{out}/textembs.pkl", "wb"))

def save_embs(df, out):
    model_name = 'bert-base-uncased'  # You can use other pre-trained BERT models as well
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = df
    text_embs = []
    pickles = 0
    # for name, group in data.groupby('link_id'):
    #     groupMessages = group.groupby('author')        
    #     for author, groupMessage in groupMessages:
    #         embs_author = []
    #         for index, row in groupMessage.iterrows():
    #             t = str(row["content"])
    #             t = t.lower()
    #             inputs = tokenizer(t, truncation=True, max_length=512, return_tensors="pt").to(device)
    #             # Get the outputs from the BERT model
    #             with torch.no_grad():
    #                 outputs = model(**inputs)
    #             # Get the embeddings (output of the [CLS] token)
    #             embeddings = outputs.last_hidden_state[:, 0, :]
    #             embs_author.append(embeddings)

    #         # here we can decide the aggregator
    #         avg = torch.mean(torch.stack(embs_author), dim=0)
    #         text_embs.append(avg)

    for author, group in data.groupby('author'):
        embs_author = []
        for index, row in group.iterrows():
            t = str(row["content"])
            t = t.lower()
            inputs = tokenizer(t, truncation=True, max_length=512, return_tensors="pt").to(device)
            # Get the outputs from the BERT model
            with torch.no_grad():
                outputs = model(**inputs)
            # Get the embeddings (output of the [CLS] token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embs_author.append(embeddings)

        # here we can decide the aggregator
        avg = torch.mean(torch.stack(embs_author), dim=0)
        text_embs.append(avg)
    print(f"Saving {len(text_embs)} embeddings")
    pickle.dump(text_embs, open(f"{out}/textembs.pkl", "wb"))

parser = argparse.ArgumentParser(description='Hypergraphs')
parser.add_argument('-s', '--size', type=int, help='Size of dataset', default=1000)
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=1000)
parser.add_argument('-o', '--optimizer', type=str, help='Optimizer choice', default='adam')
parser.add_argument('-l', '--loss', type=str, help='Loss choice', default='softmax')
args = parser.parse_args()


annotations = ['favor', 'against', 'unknown']
csv_file = "/home/ddevin/test/data/aggregated/aggregatedBefore1year.csv"
df = pd.read_csv(f"{csv_file}")
size = 1000
for i in range(1, 7):
    print(f"Dataset size: {size}")
    df_sample = df.head(size) # 32000 
    print(df_sample.describe())
    out = f"data/complete/sample{size}"
    size = size * 2
    if not os.path.exists(out):
        os.makedirs(out)
    else:
        print(f"Path {out} exists, skipping...")
        continue
    data = df_sample[df_sample.author != "AutoModerator"]
    # author,created_utc,id,link_id,subreddit,content,spinos,gemmamlabel
    data['id_num'] = data['id'].astype('category').cat.codes
    data['author_unique'] = data['author'].astype('category').cat.codes
    # data['label_id'] = data['gemmamlabel'].apply(lambda x: annotations.index(x) if x in annotations else -1)
    # data[['id_num', 'id', 'label_id']].drop_duplicates().to_csv(f"{out}/id_map.csv", index=False)

    # save hg and edge label (link_id) to file          
    with open(f"{out}/hg.hgf", 'w') as f:
        for name, group in data.groupby('link_id'):
            if len(group['link_id'].values) > 1:
                g = group['author_unique'].unique()
                g.sort()
                f.write(','.join(map(str, g)))
                f.write('\n')
            else:
                f.write(str(group['author_unique'].values[0]) + '\n')


    # # save_embeddings
    if not os.path.exists(f"{out}/textembs.pkl"):
        save_embs(data, out)
    hgToIncidenceMatrix(f"{out}/hg.hgf", f"{out}/matrix.pkl")