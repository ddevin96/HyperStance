from transformers import BertTokenizer, TFBertModel
import pandas as pd
from utils import *
import os

# bert tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")

# create all folders
create_files("data/raw", "data/processed", tokenizer, model)

# create hypergraph from the climate skeptics dataset
# and save the adjacency matrix
hgToAdjMatrix("data/processed/climateskeptics/hg.hgf", "data/processed/climateskeptics/adj.pkl")