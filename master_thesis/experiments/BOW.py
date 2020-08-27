#!/usr/bin/env python

from master_thesis.src import utils, data

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
import scipy.stats as st
import pickle


def train_BOW_model(df,
                    preprocessor = None,
                    feature_type = 'abs',
                    create_or_load = 'create',
                    max_features = 10000,
                    text_base = 'text_preprocessed',
                    target = 'avgTimeOnPagePerNr_tokens'
                    ):

    # create splits
    df_train, df_dev, df_test = data.create_train_dev_test(df = df, random_seed=123)
    print(df_train.shape, df_dev.shape, df_test.shape)

    if create_or_load == 'create':
        print("creating features", feature_type)

        # stopwords
        stopwords = nltk.corpus.stopwords.words('german')
        if preprocessor:
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

    # model: Ridge Regression
    model = Ridge()

    print("training model...")
    model.fit(X_train, y_train)

    # predict for dev set
    print("predicting dev set")
    pred_dev = model.predict(X_dev)

    # postprocessing: replace negative values with 0 (better way? can I give that hint to the model?)
    pred_dev[pred_dev < 0] = 0

    # Pearson's r as evaluation metric
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))

    # saving model with pickle
    target_path = utils.OUTPUT / 'saved_models' / f'BOW_{feature_type}_{str(max_features)}.pkl'
    pickle.dump(model, open(target_path, 'wb'))



# get raw data
df = pd.read_csv(utils.DATA / 'combined.tsv', sep = '\t')
df = df.fillna('') # replacing Nan with emtpy string
print("Shape of raw df:", df.shape)

# just take articles with ...
df = df.loc[(df['pageviews'] >= 20) &
            (df['nr_tokens'] >= 10) & # to delete articles without text or false text
            (df['avgTimeOnPagePerNr_tokens'] <= 4) &
            (df['avgTimeOnPagePerNr_tokens'] >= 0.01)
            ]
print("Remaining df after conditioning:", df.shape)

# preprocessor
preprocessor = utils.Preprocessor(delete_stopwords=True, lemmatize=True, delete_punctuation=True)

train_BOW_model(df = df,
                preprocessor = preprocessor, ############### Achtung geändert (auch saving ausgestellt)
                feature_type = 'abs',
                create_or_load ='create',
                max_features = 11000, # 100000 is to big for memory)
                text_base = 'text_preprocessed',
                target = 'avgTimeOnPagePerNr_tokens')

# to load the model
# model = pickle.load(open(target_path, 'rb'))

# TODO: note to self: without lemmatization it is even slightly better and a lot faster...

# note: bei 1000 kam ca. Pearson = .46 raus, also ganz gut
# note: bei 2000 ca. (0.3556820544408318, 1.4115783794433868e-129) also schlechter
# not: bei 1100 und mit lemmatization, MAX_NGRAM=4: (0.45406266713934623, 9.582367468030329e-220)
