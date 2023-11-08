import os
import pickle

import random

import spacy

## Load prepared dataset

directory = "dataset"

french_filename = os.path.join(directory, 'french_prepared.pkl')
english_filename = os.path.join(directory, 'english_prepared.pkl')

with open(french_filename, 'rb') as f:
    french_data = pickle.load(f)

with open(english_filename, 'rb') as f:
    english_data = pickle.load(f)

## Verify dataset

l = len(french_data)
assert l == len(english_data)

print(f"\n{l} sentences, example:")

i = random.randint(0,l-1)

print(french_data[i])
print(english_data[i])

## Tokenize sentences

# python -m spacy download fr_core_news_sm
fr = spacy.load("fr_core_news_sm") # load the French model to tokenize French text

# python -m spacy download en_core_web_sm
eng = spacy.load("en_core_web_sm") # load the English model to tokenize English text

def tokenize(text, french=True):
    """
    Tokenize a French (if french=True) or English (if french=False) text and return a list of tokens
    """
    if french:
        return [token.text for token in fr.tokenizer(text)]
    else:
        return [token.text for token in eng.tokenizer(text)]

print("\ntokenized versions:")

print(tokenize(french_data[i], french=True))
print(tokenize(english_data[i], french=False))

## Build vocabulary

def buildVocab(data, french=True):
    """
    Build and return a vocabulary (dictionary of distinct tokens) based on a list of sentences in a given language (French if french=True, else English)
    """
    vocabulary = {'unk':0}      # 'unk' being a special token, we put it at the beginning of our vocabulary
    index = 1
    for sentence in data:
        for token in tokenize(sentence, french):
            if not token in vocabulary:     # add the new token in the vocabulary
                vocabulary[token] = index
                index += 1
    return vocabulary

french_vocab = buildVocab(french_data, french=True)

print("\nfrench vocabulary length:")
print(len(french_vocab))
print("french vocabulary beginning:")
print({k: v for i, (k, v) in enumerate(french_vocab.items()) if i < 20})

english_vocab = buildVocab(english_data, french=False)

print("\nenglish vocabulary length:")
print(len(english_vocab))
print("english vocabulary beginning:")
print({k: v for i, (k, v) in enumerate(english_vocab.items()) if i < 20})

## Save vocabulary

with open(os.path.join(directory, 'french_vocab.pkl'), 'wb') as f:
    pickle.dump(french_vocab, f)

with open(os.path.join(directory, 'english_vocab.pkl'), 'wb') as f:
    pickle.dump(english_vocab, f)

print("\nVocabularies saved.")

## Numericalize sentences

# wip :

def numericalize(data, vocab, french=True):
    """
    Replace each token by its index within a list of sentences in a given language (French if french=True, else English)
    """
    new_data = [[] for _ in range(len(data))]
    for n in len(data):
        sentence = data[n]
        for token in tokenize(sentence, french):
            new_data[n].append(vocab[token])
    return new_data

french_data_num = numericalize(french_data, french_vocab, french=True)
english_data_num = numericalize(english_data, english_vocab, french=False)

print("\nnumericalized versions:")

print(french_data_num[i])
print(english_data_num[i])
