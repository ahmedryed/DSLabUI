import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import time
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization


def bert_encode(texts, tokenizer, max_len=200):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=200):
    input_word_ids = Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask,
                          segment_ids], outputs=out)
    model.compile(Adam(lr=2e-5), loss='mse', metrics=['mse'])

    return model


def finalize_model(bert_path):
    print("step 1 done")
    module = keras.models.load_model(bert_path + "\\bert_model")
    print(type(module))
    print("step 2 done")
    bert_layer = hub.KerasLayer(module_path, trainable=True)
    print("step 3 done")
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    print("step 4 done")
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    print("step 5 done")
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    print("step 6 done")
    model = build_model(bert_layer, max_len=100)
    print("step 7 done")
    model.load_weights('bert2_0checkpoint.index')
    print("step 8 done")
    return model


def predict_rating(inputText, model):
    review_encoding = bert_encode([inputText], tokenizer, max_len=100)
    return model.predict(review_encoding)
