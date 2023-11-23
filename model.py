import torch

import os
import pickle

directory = "dataset"

# Import vocabularies

with open(os.path.join(directory,'french_vocab.pkl'), 'rb') as f:
    french_vocab = pickle.load(f)

with open(os.path.join(directory,'english_vocab.pkl'), 'rb') as f:
    english_vocab = pickle.load(f)

# Import preprocessed datasets

with open(os.path.join(directory,'french_num.pkl'), 'rb') as f:
    french_sequences = pickle.load(f)

with open(os.path.join(directory,'english_num.pkl'), 'rb') as f:
    english_sequences = pickle.load(f)

