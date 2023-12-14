import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import pickle

import random

import time
import matplotlib.pyplot as plt

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

assert len(french_sequences) == len(english_sequences)

input_size = len(french_vocab)
output_size = len(english_vocab)

## Divide data into training, validation and testing sets

train_percent = 0.8
val_percent = 0.1

train_size = int(len(french_sequences) * train_percent)
val_size = int(len(french_sequences) * val_percent)
test_size = len(french_sequences) - train_size - val_size

print(f"\nNumber of training examples: {train_size}")
print(f"Number of validation examples: {val_size}")
print(f"Number of testing examples: {test_size}")

train_sequences = (torch.tensor(french_sequences[:train_size], dtype=torch.long, device=device), torch.tensor(english_sequences[:train_size], dtype=torch.long, device=device))
val_sequences = (torch.tensor(french_sequences[train_size:train_size + val_size], dtype=torch.long, device=device), torch.tensor(english_sequences[train_size:train_size + val_size], dtype=torch.long, device=device))
test_sequences = (torch.tensor(french_sequences[train_size + val_size:], dtype=torch.long, device=device), torch.tensor(english_sequences[train_size + val_size:], dtype=torch.long, device=device))

## Define training parameters and Dataloaders

# hyperparameters to adjust
embed_size = 256
hidden_size = 512
learning_rate = 0.001
n_epochs = 10
batch_size = 64

# custom datasets
train_dataset = TensorDataset(train_sequences[0], train_sequences[1])
val_dataset = TensorDataset(val_sequences[0], val_sequences[1])
test_dataset = TensorDataset(test_sequences[0], test_sequences[1])

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# criterion
pad_idx = english_vocab['pad']
crit = nn.CrossEntropyLoss(ignore_index=pad_idx)

## Define LSTM Model

class Seq2SeqLSTM(nn.Module):

    def __init__(self, inp_size, out_size, emb_size, hid_size):
        super(Seq2SeqLSTM, self).__init__()

        self.output_size = out_size

        self.enc_embedding = nn.Embedding(inp_size, emb_size)
        self.encoder = nn.LSTM(emb_size, hid_size)

        self.dec_embedding = nn.Embedding(out_size, emb_size)
        self.decoder = nn.LSTM(emb_size, hid_size)
        self.linear = nn.Linear(hid_size, out_size)

    def forward(self, source, target, tfr=0.5):

        max_len, batch_size = target.shape
        outputs = torch.zeros(max_len, batch_size, self.output_size).to(device)

        # last hidden & cell state of the encoder is used as the decoder's initial hidden state
        embedded = self.enc_embedding(source)
        _, (hidden, cell) = self.encoder(embedded)

        trg = target[0] # 'sos' token
        for i in range(1, max_len):
            embedded = self.dec_embedding(trg.unsqueeze(0))
            out, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            prediction = self.linear(out.squeeze(0))
            outputs[i] = prediction

            trg = target[i] if random.random() < tfr else prediction.argmax(1)  # either pass the next word correctly from the dataset or use the earlier predicted word (tfr = teacher forcing ratio, for training purpose only)

        return outputs

## Train and evaluate model

def evaluate(mod):
    mod.train(False)

    tot_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # turn off teacher forcing
            outputs = mod(batch[0], batch[1], tfr=0) 
            outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
            trg_flatten = batch[1][1:].view(-1)

            loss = crit(outputs_flatten, trg_flatten)
            tot_loss += loss.item()

    return tot_loss / len(val_loader)

def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):  
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        start_time = time.time()

        mod.train(True)

        epoch_loss = 0
        for batch in train_loader:
            optim.zero_grad()

            outputs = mod(batch[0], batch[1])
            outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
            trg_flatten = batch[1][1:].view(-1)

            loss = crit(outputs_flatten, trg_flatten)
            epoch_loss += loss.item()
            loss.backward()

            optim.step()
        
        train_loss = epoch_loss / len(train_loader)
        print(f"Train Loss: {(epoch_loss / len(train_loader)):.4f}")

        val_loss = evaluate(mod)
        print(f"Validation Loss: {val_loss:.4f}")
        
        end_time = time.time()
        print(f"Time: {(end_time-start_time):.2f}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(1, figsize=(10,5))

    plt.plot(list(range(1, n_epochs+1)), train_losses, 'r', label="train")
    plt.plot(list(range(1, n_epochs+1)), val_losses, 'b', label="val")

    plt.ylabel("Cross-entropy loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    plt.show()

## main

model = Seq2SeqLSTM(input_size, output_size, embed_size, hidden_size).to(device)

train(model)