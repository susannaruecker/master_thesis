#!/usr/bin/env python

from master_thesis.src import utils, data

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

#import scipy.stats as st
import pickle


def train_BOW_model(df,
                    preprocessor = None,
                    feature_type = 'abs',
                    create_or_load = 'create',
                    max_features = 10000,
                    text_base = 'text_preprocessed',
                    target = 'time_class'
                    ):

    # create splits
    df_train, df_dev, df_test = data.create_train_dev_test(df = df, random_seed=123)
    print(df_train.shape, df_dev.shape, df_test.shape)

    if create_or_load == 'create':
        print("creating features", feature_type)

        # stopwords
        stopwords = nltk.corpus.stopwords.words('german')
        if preprocessor: #TODO: Das ist dämlich! So wird die Stopwordliste gerade bei delete_stopwords == True eliminiert!
                         # andererseits entfernt der preprocessor die stopwörter sowieso also ist es vielleicht auch egal
            stopwords = [ preprocessor(s) for s in stopwords ]
        print(stopwords[:10])

        # instantiate countVectorizer
        MAX_NGRAM = 4
        MIN_DF = 5

        vectorizer = CountVectorizer(analyzer='word',
                                     preprocessor= preprocessor,
                                     ngram_range= (1, MAX_NGRAM),
                                     min_df = MIN_DF, # Achtung: vielleicht je nach Textbasis anpassen?
                                     max_features = max_features,
                                     stop_words = stopwords
                                    )

        print("fitting vectorizer and transforming texts to features...")
        X_train = vectorizer.fit_transform(df_train[text_base])
        X_dev = vectorizer.transform(df_dev[text_base])
        X_test = vectorizer.transform(df_test[text_base])

        X_train, X_dev, X_test = X_train.toarray(), X_dev.toarray(), X_test.toarray()

        np.save(utils.OUTPUT / 'BOW_features' / f'X_train_abs_{str(max_features)}.npy', X_train)
        np.save(utils.OUTPUT / 'BOW_features' / f'X_dev_abs_{str(max_features)}.npy', X_dev)
        np.save(utils.OUTPUT / 'BOW_features' / f'X_test_abs_{str(max_features)}.npy', X_test)

        if feature_type == 'abs':
            X_train, X_dev, X_test = X_train, X_dev, X_test

        if feature_type == 'rel':
            # convert to relative frequencies #TODO: Achtung, hier kann Division durch 0 auftreten, wie damit umgehen?
            X_train_rel = X_train/X_train.sum(axis=1, keepdims=True)
            X_dev_rel = X_dev/X_dev.sum(axis=1, keepdims=True)
            X_test_rel = X_test/X_test.sum(axis=1, keepdims=True)

            np.save(utils.OUTPUT / 'BOW_features' / f'X_train_rel_{str(max_features)}.npy', X_train_rel)
            np.save(utils.OUTPUT / 'BOW_features' / f'X_dev_rel_{str(max_features)}.npy', X_dev_rel)
            np.save(utils.OUTPUT / 'BOW_features' / f'X_test_rel_{str(max_features)}.npy', X_test_rel)

            X_train, X_dev, X_test = X_train_rel, X_dev_rel, X_test_rel

        if feature_type == 'bool':
            # convert to binary values
            X_train_bool = X_train > 0
            X_dev_bool = X_dev > 0
            X_test_bool = X_test > 0

            np.save(utils.OUTPUT / 'BOW_features' / f'X_train_bool_{str(max_features)}.npy', X_train_bool)
            np.save(utils.OUTPUT / 'BOW_features' / f'X_dev_bool_{str(max_features)}.npy', X_dev_bool)
            np.save(utils.OUTPUT / 'BOW_features' / f'X_test_bool_{str(max_features)}.npy', X_test_bool)

            X_train, X_dev, X_test = X_train_bool, X_dev_bool, X_test_bool

    if create_or_load == 'load':
        print("loading features...", feature_type)
        X_train = np.load(utils.OUTPUT / 'BOW_features' / f'X_train_{feature_type}_{str(max_features)}.npy')
        X_dev = np.load(utils.OUTPUT / 'BOW_features' / f'X_dev_{feature_type}_{str(max_features)}.npy')
        X_test = np.load(utils.OUTPUT / 'BOW_features' / f'X_test_{feature_type}_{str(max_features)}.npy')

    print("Feature shapes: ", X_train.shape, X_dev.shape, X_test.shape)

    # define labels
    y_train = np.array(df_train[target])
    y_dev = np.array(df_dev[target])
    y_test = np.array(df_test[target])

    print("Labels shape: ", y_train.shape, y_dev.shape, y_test.shape)

    # for classification
    model = LogisticRegression(max_iter=1000)

    print("training model...")
    model.fit(X_train, y_train)

    # predict for dev set
    print("predicting dev set")
    pred_dev = model.predict(X_dev)

    # evaluating
    accuracy = accuracy_score(y_dev, pred_dev) # overall
    print("overall accuracy:", accuracy)
    print(classification_report(y_dev, pred_dev)) #, output_dict=True))


# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()
df = df[['text_preprocessed', 'time_class']] # to save space
print(df.head())

# preprocessor
preprocessor = utils.Preprocessor(delete_stopwords=False, lemmatize=True, delete_punctuation=True)

train_BOW_model(df = df,
                preprocessor = preprocessor,
                feature_type = 'abs',
                create_or_load ='create',
                max_features = 1000, # 100000 is to big for memory)
                text_base = 'text_preprocessed',
                target = 'time_class')

# bei 500: acc = 0.57
# bei 1000: acc = 0.56