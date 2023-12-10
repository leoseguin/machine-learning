import string
import re
from pickle import dump
from unicodedata import normalize
 
from pickle import load
from collections import Counter

import requests
import tarfile
import os

### This program downloads the dataset from the web, then prepares the data for learning, and saves it to the computer under dataset/french_prepared.pkl and dataset/english_prepared.pkl. It needs an internet connection to run properly and can take time to process, but it has to be run only once.

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_sentences(doc):
    return doc.strip().split('\n')

# shortest and longest sentence lengths
def sentence_lengths(sentences):
    lengths = [len(s.split()) for s in sentences]
    return min(lengths), max(lengths)

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab

# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurance):
	tokens = [k for k,c in vocab.items() if c >= min_occurance]
	return set(tokens)

# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)
	return new_lines

def clean_dataset(dir):
	
	## Split the files into sentences

    # load English data
    filename = os.path.join(dir, 'europarl-v7.fr-en.en')
    doc = load_doc(filename)
    sentences = to_sentences(doc)
    minlen, maxlen = sentence_lengths(sentences)
    print('English data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))
	
    # load French data
    filename = os.path.join(dir, 'europarl-v7.fr-en.fr')
    doc = load_doc(filename)
    sentences = to_sentences(doc)
    minlen, maxlen = sentence_lengths(sentences)
    print('French data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))

    ## Normalize and clean files

    # load English data
    filename = os.path.join(dir, 'europarl-v7.fr-en.en')
    doc = load_doc(filename)
    sentences = to_sentences(doc)
    sentences = clean_lines(sentences)
    filename = os.path.join(dir, 'english.pkl')
    save_clean_sentences(sentences, filename)
    """
    # spot check
    for i in range(10):
        print(sentences[i])
    """
	
    # load French data
    filename = os.path.join(dir, 'europarl-v7.fr-en.fr')
    doc = load_doc(filename)
    sentences = to_sentences(doc)
    sentences = clean_lines(sentences)
    filename = os.path.join(dir, 'french.pkl')
    save_clean_sentences(sentences, filename)
    """
    # spot check
    for i in range(10):
        print(sentences[i])
	"""

    ## Reduce vocabulary

    # load English dataset
    filename = os.path.join(dir, 'english.pkl')
    lines = load_clean_sentences(filename)
    # calculate vocabulary
    vocab = to_vocab(lines)
    print('English Vocabulary: %d' % len(vocab))
    # reduce vocabulary
    vocab = trim_vocab(vocab, 50)
    print('New English Vocabulary: %d' % len(vocab))
    # mark out of vocabulary words
    lines = update_dataset(lines, vocab)
    # save updated dataset
    filename = os.path.join(dir, 'english_prepared.pkl')
    save_clean_sentences(lines, filename)
    """
	# spot check
    for i in range(10):
        print(lines[i])
	"""

    # load French dataset
    filename = os.path.join(dir, 'french.pkl')
    lines = load_clean_sentences(filename)
    # calculate vocabulary
    vocab = to_vocab(lines)
    print('French Vocabulary: %d' % len(vocab))
    # reduce vocabulary
    vocab = trim_vocab(vocab, 50)
    print('New French Vocabulary: %d' % len(vocab))
    # mark out of vocabulary words
    lines = update_dataset(lines, vocab)
    # save updated dataset
    filename = os.path.join(dir, 'french_prepared.pkl')
    save_clean_sentences(lines, filename)
    """
	# spot check
    for i in range(10):
        print(lines[i])
	"""
	
## main

download_dir = 'dataset'
os.makedirs(download_dir, exist_ok=True)

url = 'http://www.statmt.org/europarl/v7/fr-en.tgz'
filename = os.path.join(download_dir, os.path.basename(url))

print("Download starting...")
response = requests.get(url)

if response.status_code == 200:  
    # Download the dataset
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

    # Unzip the .tgz file
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=download_dir)
    print(f"Unzipped {filename}\n")

    # Clean the dataset
    print("Dataset cleaning...")
    clean_dataset(download_dir)
    print("Dataset cleaned\n")

else:
    print(f"Failed to download {url}. Status code: {response.status_code}")