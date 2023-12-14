# Machine learning mini-project : French-English translation

*Report by **[Léo Seguin](https://github.com/leoseguin)***

## Introduction

Language translation plays a vital role in our society, by easing access to information in diverse languages. This project focuses on the development of a **machine translation model** to perform **French-to-English** translation tasks. For this purpose, we will create a model in PyTorch and train it on a prepared extract of the public *Europarl* dataset.

This report discusses the methodology used for the project, including the dataset used for training and evaluation, the neural network model employed, and the outcomes of our experiments. 

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing step](#preprocessing-step)
- [Model architecture](#model-architecture)
- [Hyperparameters optimisation](#hyperparameters-optimisation)

## Dataset

*Note: for this chapter, we took our main inspiration (and Python code) from [this website](https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/).*

The data used in the project is taken from the public *Europarl* dataset. It is the contraction of *European Parliament*, as it contains a collection of transcriptions of speakers at the European Parliament, between 1996 and 2011. Those transcriptions are translated in the 11 official languages of the EU.

In this project, we focus only on the French and English transcriptions. 
They are stored in two parallel datasets, `europarl-v7.fr-en.fr` and `europarl-v7.fr-en.en`. They contain the same **2,007,723 sentences**, respectively in their French and English versions. The French dataset contains 51,388,643 words (of which 141,642 are distinct), while the English datasets contains 50,196,035 of them (of which 105,357 are distinct).

We follow different steps in order to prepare the data for a more efficient learning, notably: splitting the file into sentences, tokenizing text by white space, normalizing case to lowercase, removing punctuation from each word, removing non-printable characters, converting French characters to Latin characters, removing words that contain non-alphabetic characters, and reducing vocabulary (most infrequent words are replaced by "unk" in the final dataset). 
The prepared data is stored in the two parallel datasets `french_prepared.pkl` and `english_prepared.pkl`. They still contain the same 2,007,723 sentences; however, only **32,871 distinct French words** and **23,340 distinct English words** are kept.

The program [`load_dataset.py`](load_dataset.py) downloads the dataset from the web, applies the cleaning steps mentioned above, and saves the resulting files in the `dataset` folder.

## Preprocessing step

*Note: for this chapter, we took our main inspiration from [this website](https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html).*

We follow another series of steps in order to put the data in a format suitable for learning, i.e. lists of numbers of equal length, and significantly improve performance. The steps are the following:
- Tokenize sentences
- Remove the 5% shortest (resp. 40% longest) sentences, in order to increase the minimum sentence length from 0 token to 6 (resp. decrease the maximum from 642 tokens to 27)
- Randomly remove 99.5% of the remaining sentences
- Build a vocabulary for each language
- For each sentence in the dataset, add start-of-sentence and end-of-sentence tokens and replace every token by its index in the corresponding vocabulary
- Add padding tokens at the end of sequences so that all of them have the same length
- Save the vocabularies and the modified datasets in new pickle files

The vocabularies are saved under `french_vocab.pkl` and `english_vocab.pkl`; they contain respectively **9,857 French tokens** and **7,494 English tokens**. The prepared data is stored in the two parallel datasets `french_num.pkl` and `english_num.pkl`, each one containing **5,419 lists of 29 integers each**, ready for machine learning application.

For instance, the sentence `"we support free choice for all consumers across the eu"` is now encoded as `[2, 7, 76, 1230, 472, 77, 192, 155, 1537, 16, 302, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` in the file `english_num.pkl`.

The program [`preprocessing.py`](preprocessing.py) applies the preprocessing steps mentioned above, and saves the resulting files in the `dataset` folder.

## Model architecture

*Note: for this chapter, we took our main inspiration (and some code snippets) from the three following websites [[1]](https://cnvrg.io/seq2seq-model/), [[2]](https://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html#Encoder-Module), [[3]](https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350#6756).*

We choose to use a **Seq2Seq (Sequence to Sequence)** architecture with **LSTM (Long Short-Term Memory)**.
- Seq2Seq models are particularly adept at language translation tasks and are widely used in this domain.
- LSTMs are an extension of RNNs (Recurrent Neural Networks) designed to better handle some of the issues encountered by traditional RNNs, notably the vanishing or exploding gradient problem. They can handle sequences of varying lengths and better memorize information over long sequences. For translation, the ability to understand contextual information and memorize it through variable-length sentences is crucial.

The Seq2Seq model is comprised of an encoder-decoder framework:
- The encoder turns the input (here, a sentence in French) into a context vector that captures the input sequence's semantic meaning.
- The decoder uses the context vector generated by the encoder to produce an output sentence in the target language (here, English).

Below are some interesting considerations about the model architecture:
- We divide the data into training, validation, and test sets. An **80%** proportion is allocated **for training**, **10% for validation**, and the remaining **10% for testing**.
- The chosen loss function is **cross-entropy** (`nn.CrossEntropyLoss`), which is suitable for evaluating the difference between the model's prediction and the ground truth in a sentence translation context.
- The chosen optimizer is **Adam** (`torch.optim.Adam`), which is a popular and efficient optimization algorithm, commonly used for machine translation tasks.
- We divide the dataset into **mini-batches** using `DataLoader`. They allow training the model in smaller portions of samples at a time, which helps to reduce memory load and speed up learning.
- We use **embeddings** (`nn.Embedding`) to help computing the context vectors mentioned above. They allow to reduce the dimension of the input and output data, equal to the vocabulary size (respectively `input_size` and `output_size`), to a chosen, lower dimension (`embed_size`), before feeding it to the LSTMs.
- During training, we employ **teacher forcing**, a technique where the model sometimes uses the ground truth tokens (with a probability `tfr`) instead of its own predictions. This helps to stabilize and accelerate the convergence of the model and limits the propagation of errors.

The program [`model.py`](model.py) creates the model and trains it on the data contained in the `dataset` folder.

## Hyperparameters optimisation

wip

## TODO :
1. try and optimise hyperparameters
2. save the model when ready and compute loss on test set
3. try some translation usages (compare with Google Translate and/or DeepL)