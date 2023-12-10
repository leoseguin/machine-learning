import os
import pickle

import random

import spacy

import numpy as np

### This program applies the preprocessing step to the prepared data generated by load_dataset.py. It can take time to process, but it has to be run only once.

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

def tokenize(data, french=True):
    """
    Take as input a French (if french=True) or English (if french=False) list of sentences (strings) and return the list of tokenized sentences (lists of tokens)
    """
    if french:
        return [[token.text for token in fr.tokenizer(data[k])] for k in range(len(data))]
    else:
        return [[token.text for token in eng.tokenizer(data[k])] for k in range(len(data))]

english_token = tokenize(english_data, french=False)
french_token = tokenize(french_data, french=True)

## Remove the shortest and longest sentences

min_length = min( min([len(sentence) for sentence in french_token]), min([len(sentence) for sentence in english_token]) )  # length of the shortest sentence in both datasets
print(f"\nLowest sentence length: {min_length}")
max_length = max( max([len(sentence) for sentence in french_token]), max([len(sentence) for sentence in english_token]) )  # length of the longest sentence in both datasets
print(f"Highest sentence length: {max_length}")

percentile_low = 10
fr_percentile_low = int(np.percentile([len(sentence) for sentence in french_token], percentile_low))
en_percentile_low = int(np.percentile([len(sentence) for sentence in english_token], percentile_low))
min_length = min(fr_percentile_low, en_percentile_low)
print(f"Limit minimum sentence length (top {percentile_low}%): {min_length}")

percentile_high = 60
fr_percentile_high = int(np.percentile([len(sentence) for sentence in french_token], percentile_high))
en_percentile_high = int(np.percentile([len(sentence) for sentence in english_token], percentile_high))
max_length = max(fr_percentile_high, en_percentile_high)
print(f"Limit maximum sentence length (top {percentile_high}%): {max_length}")

def removeShortestAndLongestSentences(fr_data, en_data, min_len, max_len):
    """
    Remove the sentences that are longer than max_len from their dataset, along with their translation in the other dataset
    """
    assert len(fr_data) == len(en_data)
    fr_filtered = []
    en_filtered = []
    for fr_sent, en_sent in zip(fr_data, en_data):
        if min_len <= len(fr_sent) <= max_len and min_len <= len(en_sent) <= max_len:
            fr_filtered.append(fr_sent)
            en_filtered.append(en_sent)
    return fr_filtered, en_filtered

french_token, english_token = removeShortestAndLongestSentences(french_token, english_token, min_length, max_length)

print(f"\nTop {percentile_low}% shortest and {100-percentile_high}% longest sentences removed. {len(french_token)} remaining sentences.")

## Randomly reduce the number of sentences in the dataset

def reduceDataset(data, p):
    """
    Randomly remove a percentage p of the sentences contained in the data list
    """
    random.seed(0)
    num_to_remove = int(len(data) * (p / 100))  # number of sentences to remove
    indices_to_keep = random.sample(range(len(data)), len(data) - num_to_remove)
    return [data[i] for i in indices_to_keep]

percent_to_remove = 99
french_token, english_token = reduceDataset(french_token, percent_to_remove), reduceDataset(english_token, percent_to_remove)

print(f"\n{percent_to_remove}% additional sentences randomly removed. {len(french_token)} remaining sentences.")

## Build vocabulary

def buildVocab(data):
    """
    Build and return a vocabulary (dictionary of distinct tokens) based on a list of sentences in a given language (French if french=True, else English)
    """
    vocabulary = {'pad':0, 'unk':1}      # 'pad' and 'unk' being special tokens, we put it at the beginning of our vocabulary. 'pad' refers to the padding added at the end of sentences while 'unk' refers to the words that were removed from the vocabulary.
    index = 2
    for sentence in data:
        for token in sentence:
            if not token in vocabulary:     # add the new token in the vocabulary
                vocabulary[token] = index
                index += 1
    return vocabulary

french_vocab = buildVocab(french_token)

print("\nFrench vocabulary:")
print(f"{len(french_vocab)} tokens, beginning:")
print({k: v for j, (k, v) in enumerate(french_vocab.items()) if j < 20})

english_vocab = buildVocab(english_token)

print("\nEnglish vocabulary:")
print(f"{len(english_vocab)} tokens, beginning:")
print({k: v for j, (k, v) in enumerate(english_vocab.items()) if j < 20})

## Save vocabulary

with open(os.path.join(directory, 'french_vocab.pkl'), 'wb') as f:
    pickle.dump(french_vocab, f)

with open(os.path.join(directory, 'english_vocab.pkl'), 'wb') as f:
    pickle.dump(english_vocab, f)

print("\nVocabularies saved.")

## Numericalize sentences

i = random.randint(0, len(french_token)-1)

print("\nExample tokenized sentence:")

print(french_token[i])
print(english_token[i])

def numericalize(data, vocab):
    """
    Replace each token by its index within a list of sentences in a given language (French if french=True, else English)
    """
    new_data = [[] for _ in range(len(data))]
    for n in range(len(data)):
        for token in data[n]:
            new_data[n].append(vocab[token])
    return new_data

french_data_num = numericalize(french_token, french_vocab)
english_data_num = numericalize(english_token, english_vocab)

print("\nSentences succesfully numericalized. Same example:")

print(french_data_num[i])
print(english_data_num[i])

## Apply padding

def applyPadding(fr_data, en_data, max_len):
    """
    Add padding tokens at the end of the sentences in both datasets so that all of them have the same length
    """
    assert len(fr_data) == len(en_data)
    for n in range(len(fr_data)):
        fr_sentence, en_sentence = fr_data[n], en_data[n]
        fr_sentence += [0] * (max_len - len(fr_sentence))
        en_sentence += [0] * (max_len - len(en_sentence))
        assert len(fr_sentence) == max_len and len(en_sentence) == max_len
    return fr_data, en_data

french_data_num, english_data_num = applyPadding(french_data_num, english_data_num, max_length)

print(f"\nPadding added to all sentences with less than {max_length} words. Same example:")

print(french_data_num[i])
print(english_data_num[i])

## Save numericalized sentences

with open(os.path.join(directory, 'french_num.pkl'), 'wb') as f:
    pickle.dump(french_data_num, f)

with open(os.path.join(directory, 'english_num.pkl'), 'wb') as f:
    pickle.dump(english_data_num, f)

print("\nNumericalized datasets saved.")