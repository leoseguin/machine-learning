import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import pickle

import numpy as np

directory = "dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Import prepared datasets and vocabularies

with open(os.path.join(directory,'french_num.pkl'), 'rb') as f:
    french_sequences = pickle.load(f)
with open(os.path.join(directory,'english_num.pkl'), 'rb') as f:
    english_sequences = pickle.load(f)

with open(os.path.join(directory,'french_vocab.pkl'), 'rb') as f:
    french_vocab = pickle.load(f)
with open(os.path.join(directory,'english_vocab.pkl'), 'rb') as f:
    english_vocab = pickle.load(f)

## Divide data into training, validation and testing sets

assert len(french_sequences) == len(english_sequences)

train_percent = 0.8
val_percent = 0.1

train_size = int(len(french_sequences) * train_percent)
val_size = int(len(french_sequences) * val_percent)
test_size = len(french_sequences) - train_size - val_size

print(f"\nNumber of training examples: {train_size}")
print(f"Number of validation examples: {val_size}")
print(f"Number of testing examples: {test_size}\n")

train_sequences = (french_sequences[:train_size], english_sequences[:train_size])
val_sequences = (french_sequences[train_size:train_size + val_size], english_sequences[train_size:train_size + val_size])
test_sequences = (french_sequences[train_size + val_size:], english_sequences[train_size + val_size:])

## Make batches

def make_batch(sequences):
    french_seqs, english_seqs = sequences
    input_batch, output_batch, target_batch = [], [], []

    for i in range(len(french_seqs)):
        input = french_seqs[i]
        output = english_seqs[i]
        target = english_seqs[i]

        input_batch.append(np.eye(len(french_vocab))[input])
        output_batch.append(np.eye(len(english_vocab))[output])
        target_batch.append(target)     # not one-hot

    return torch.tensor(input_batch, dtype=torch.long, device=device), torch.tensor(output_batch, dtype=torch.long, device=device), torch.tensor(target_batch, dtype=torch.long, device=device)

train_batches = make_batch(train_sequences)
val_batches = make_batch(val_sequences)
test_batches = make_batch(test_sequences)

## Define DataLoaders and training parameters
"""
# custom datasets
train_dataset = TensorDataset(train_batches[0], train_batches[1], train_batches[2])
val_dataset = TensorDataset(val_batches[0], val_batches[1], val_batches[2])
test_dataset = TensorDataset(test_batches[0], test_batches[1], test_batches[2])

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
"""
# hyperparameters to adjust
hidden_size = 256
learning_rate = 0.001
n_epochs = 10
batch_size = 64

# etc
crit = nn.CrossEntropyLoss()
input_size = len(french_sequences[0])

## Define LSTM Model

class Seq2SeqLSTM(nn.Module):

    def __init__(self, inp_size, hid_size):
        super(Seq2SeqLSTM, self).__init__()

        self.hidden_size = hid_size
        self.encoder = nn.LSTM(inp_size, hid_size, dropout=0.5)
        self.decoder = nn.LSTM(inp_size, hid_size, dropout=0.5)
        self.linear = nn.Linear(hid_size, inp_size)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)

        _, enc_states = self.encoder(enc_input, enc_hidden)
        outputs, _ = self.decoder(dec_input, enc_states)

        model = self.linear(outputs)
        return model

## Train and evaluate model

def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        mod.train(True)

        hidden = torch.zeros(1, batch_size, hidden_size)

        optim.zero_grad()

        input_batch = train_batches[0].to(device)
        output_batch = train_batches[1].to(device)
        target_batch = train_batches[2].to(device)

        output = mod(input_batch, hidden, output_batch)
        output = output.transpose(0, 1)

        loss = 0.0
        for i in range(len(target_batch)):
            
            loss += crit(output[i], target_batch[i])

        loss.backward()
        optim.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

## main

model = Seq2SeqLSTM(input_size, hidden_size).to(device)

train(model)