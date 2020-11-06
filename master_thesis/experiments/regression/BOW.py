#!/usr/bin/env python

from master_thesis.src import utils, data

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
import scipy.stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle


def train_BOW_model(df,
                    preprocessor = None,
                    feature_type = 'abs',
                    create_or_load = 'create',
                    max_features = 1000,
                    text_base = 'textPublisher_preprocessed',
                    target = 'avgTimeOnPagePerWordcount'
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
        MAX_NGRAM = 5
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

    # Pearson's r and MSE as evaluation metric
    print("text_base:", text_base)
    print("target:", target)
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))
    print("MSE: ", mean_squared_error(pred_dev, y_dev))
    print("MAE: ", mean_absolute_error(pred_dev, y_dev))

    # print some predictions
    print( [ p.round(2) for p in y_dev[:50] ])
    print( [ p.round(2) for p in pred_dev[:50] ])

    # saving model with pickle
    target_path = utils.OUTPUT / 'saved_models' / f'BOW_{feature_type}_{str(max_features)}.pkl'
    pickle.dump(model, open(target_path, 'wb'))


# get data (already conditionend on min_pageviews etc)
df = utils.get_raw_df()
df = df[df.txtExists == True]
df = df[df.nr_tokens_publisher >= 70]
df = df[df.zeilen >= 10]
print(df.shape)

# preprocessor
preprocessor = utils.Preprocessor(delete_stopwords=False, lemmatize=True, delete_punctuation=False)

train_BOW_model(df = df,
                preprocessor = preprocessor,
                feature_type = 'abs',
                create_or_load ='create',
                max_features = 300, #TODO: warum ist hier so wenig deutlich besser als zB 1000?
                text_base = 'textPublisher_preprocessed', #'teaser', #'text_preprocessed',
                target = 'avgTimeOnPagePerWordcount' #'stickiness' #avgTimeOnPage' #'avgTimeOnPagePerRow' # 'avgTimeOnPagePerWordcount'
                )

# to load the model
# model = pickle.load(open(target_path, 'rb'))


# bei max_features = 300, Pearson 0.666, MAE 0.187, MSE 0.072