import torch
import pickle

# Load dataset

with open('french_vocab.pkl', 'rb') as f:
    french_data = pickle.load(f)

with open('english_vocab.pkl', 'rb') as f:
    english_data = pickle.load(f)