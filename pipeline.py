# import warnings
# warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)
from transformers import BertTokenizer, TFBertModel
import pandas as pd
from utils import *
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")

create_files("data/raw", "data/processed", tokenizer, model)