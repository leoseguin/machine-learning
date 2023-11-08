# Machine learning mini-project : French-English translation

Report by **[Léo Seguin](https://github.com/leoseguin)**

## Introduction

Language translation plays a vital role in our society, by easing access to information in diverse languages. This project focuses on the development of a **machine translation model** to perform **English-to-French** and **French-to-English** translation tasks. For this purpose, we will create a model in PyTorch and train it on the prepared *Europarl* dataset defined [here](https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/).

This report discusses the methodology used for the project, including the dataset used for training and evaluation, the neural network model employed, and the outcomes of our experiments. 

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing step](#preprocessing-step)

## Dataset

The dataset used in the project is taken from the public *Europarl* dataset. It is the contraction of *European Parliament*, as it contains a collection of transcriptions of speakers at the European Parliament, between 1996 and 2011. Those transcriptions are translated in the 11 official languages of the EU.

In this project, we focus only on the French and English transcriptions. 
They are stored in two parallel datasets, `europarl-v7.fr-en.fr` and `europarl-v7.fr-en.en`. They contain the same **2,007,723 sentences**, respectively in their French and English versions. The French dataset contains 51,388,643 words (of which 141,642 are distinct), while the English datasets contains 50,196,035 of them (of which 105,357 are distinct).

We follow different steps in order to prepare the data for a more efficient learning, notably: splitting the file into sentences, tokenizing text by white space, normalizing case to lowercase, removing punctuation from each word, removing non-printable characters, converting French characters to Latin characters, removing words that contain non-alphabetic characters, and reducing vocabulary (all removed words are replaced by "unk" in the final dataset). 
The prepared data is stored in the two parallel datasets `french_prepared.pkl` and `english_prepared.pkl`. They still contain the same 2,007,723 sentences; however, only **58,800 distinct French words** and **41,746 distinct English words** are kept.

The program [`load_dataset.py`](load_dataset.py) downloads the dataset from the web, applies the preparation steps, and saves the results in the `dataset` folder.

## Preprocessing step

wip

https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html

- Tokenize sentences
- Build a vocabulary for each language
- Replace each token by its index in the sentences from the dataset
- Save the vocabularies and the modified datasets in new pickle files