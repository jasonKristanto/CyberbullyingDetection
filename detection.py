import pandas as pd
import re
import keras
import numpy as np
import time
import nltk
import tensorflow as tf

from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing import text
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model, Sequential, load_model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense

from gensim.models.fasttext import FastText


def tokenization(dataTeks):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=1439, split=" ")
    tokenizer.fit_on_texts(dataTeks.values)

    word_index = tokenizer.word_index

    X = tokenizer.texts_to_sequences(dataTeks.values)
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=25)

    return X, word_index


def textPreprocessing(dataTeks):
    dataTeks = dataTeks.apply(lambda x: x.lower())
    dataTeks = dataTeks.apply(lambda x: re.sub('\\\\"', ' ', x))
    dataTeks = dataTeks.apply(lambda x: re.sub('[^a-zA-Z0-9\s]', ' ', x))
    dataTeks = dataTeks.apply(lambda x: re.sub('\s+', ' ', x))
    dataTeks = dataTeks.apply(lambda x: x.strip())

    data, word_index = tokenization(dataTeks)

    return data, word_index


def prediction(pred):
    threshold = 0.5

    hasil_predict = [""] * (len(pred) + 1)

    for i in range(0, len(pred)):
        if pred[i] >= threshold:
            hasil_predict[i] = "Cyberbullying"
        else:
            hasil_predict[i] = "Not Cyberbullying"      

    return hasil_predict


def fasttext(word_index):
    # get the vectors
    loaded_ft = FastText.load("ft_model_100_andriansyah_defaultconfig.bin")

    embedding_matrix = np.zeros((len(word_index)+1, 100))
    word_not_found = []
    for word, i in word_index.items():
        if word in loaded_ft.wv:
            embedding_vector = loaded_ft[word]
            # words that cannot be found will be set to 0
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                word_not_found.append(word)
        else:
            word_not_found.append(word)
            

    return embedding_matrix;


global graph
graph = tf.get_default_graph()

epoch = 10
batch_size = 8
unit = 25
dropout = 0.05
regularization = 0.001
activation = 'sigmoid'
optimizer = 'Adadelta'

def cyberbullying_detection(dataTeks):
    dataTeks = pd.DataFrame(dataTeks)

    data, word_index = textPreprocessing(dataTeks[0])
    
    embedding_matrix = fasttext(word_index)

    with graph.as_default():
        model = Sequential()
        model.add(Embedding(len(word_index)+1, 100, input_length=data.shape[1], 
                            weights=[embedding_matrix], trainable=False))
        
        model.add(LSTM(unit, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, 
                    kernel_regularizer=keras.regularizers.l2(regularization), name='lstm_3'))
        model.add(LSTM(unit, dropout=dropout, recurrent_dropout=dropout, 
                    kernel_regularizer=keras.regularizers.l2(regularization), name='lstm_4'))
        
        model.add(Dense(1, activation=activation, name='dense_2'))

        model.load_weights('model_cyberbullying_detection_100embeddingsize.h5', by_name='lstm_3')
        model.load_weights('model_cyberbullying_detection_100embeddingsize.h5', by_name='lstm_4')
        model.load_weights('model_cyberbullying_detection_100embeddingsize.h5', by_name='dense_2')

        model.compile(loss='binary_crossentropy', optimizer=optimizer)

        pred = model.predict(data)

    hasil_predict = prediction(pred)

    result = dict()
    ctr = 0

    for hasil in hasil_predict:
        result[ctr] = hasil
        ctr += 1

    return result
