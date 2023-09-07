# Automatic Text Generation with LSTM RNN

This Jupyter notebook demonstrates text generation using a Long Short-Term Memory (LSTM) Recurrent Neural Network model trained on a corpus of Amazon product reviews.

## Overview

The notebook loads a dataset of 5000 Amazon reviews for a Fire HD tablet. It performs the following steps:

1. Imports libraries like Pandas, Numpy, and Keras.

2. Loads the review dataset into a Pandas DataFrame.

3. Plots the number of reviews over time to visualize the data. 

4. Tokenizes the text data into words and sentences using NLTK.

5. Loads a pre-trained Word2Vec embedding model from Google News data.

6. Creates training sequences and labels by extracting subsequences from each review.

7. Converts the words to indexes based on the vocabulary. 

8. Creates an embedding matrix using Word2Vec vectors for each word.

9. Builds a Keras LSTM model with trained embedding layer, LSTM layer, dropout, and dense layer.

10. Compiles and trains the model on the review data.

11. Generates new text by feeding seed text into the trained model to predict next words.

## Model Details

The model architecture consists of:

- Embedding layer initialized with Word2Vec embeddings
- Masking layer 
- LSTM layer with 50 memory units
- Dropout layer for regularization
- Fully connected output layer with softmax activation

The model is trained for 25 epochs with early stopping to prevent overfitting. Checkpointing is used to save the best model weights.

## Usage

The notebook provides full code to load data, build, train and generate text from the LSTM model.

It can be easily adapted to any text corpus by:

- Changing the input data source
- Adjusting model hyperparameters
- Modifying text preprocessing

The resulting model can generate new realistic text after training on a large dataset.

## References

The notebook implements text generation based on concepts from:

- Embedding words for input to RNN: https://towardsdatascience.com/word-embedding-for-sequence-modeling-part-1-e585c43994db
- LSTM for text generation: https://towardsdatascience.com/text-generation-with-lstm-recurse-networks-8565c2402f28
- Text generation using Keras LSTM: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
