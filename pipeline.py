import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import logging
logging.set_verbosity_error()

from transformers import BertTokenizer, TFBertModel
import pandas as pd
from utils import *
import os

## bert tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")

## create all processed data for each subreddit
create_files("data/raw", "data/processed", tokenizer, model)