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
import json


def train_BOW_model(publisher = 'NOZ',
                    preprocessor = None,
                    feature_type = 'abs',
                    create_or_load = 'create',
                    max_features = 500,
                    text_base = 'article_text',
                    target = 'avgTimeOnPage'
                    ):

    df = utils.get_publisher_df(publisher)

    with open(utils.OUTPUT / "splits" / f"{publisher}_splits.json", "r") as f:
        splits = json.load(f)
        train_IDs = splits["train"]
        dev_IDs = splits["dev"]
        test_IDs = splits["test"]

    df_train = df.loc[train_IDs]
    df_dev = df.loc[dev_IDs]
    df_test = df.loc[test_IDs]

    print(df_train.shape, df_dev.shape, df_test.shape)

    if create_or_load == 'create':
        print("creating features", feature_type)
        print("max_features = ", max_features)

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

        np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_abs_{str(max_features)}.npy', X_train)
        np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_abs_{str(max_features)}.npy', X_dev)
        np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_abs_{str(max_features)}.npy', X_test)

        # saving vectorizer (for inference and for having feature_names)
        with open(utils.OUTPUT / 'BOW_features' / f'{publisher}_vectorizer_{str(max_features)}.pkl', "wb") as fp:
            pickle.dump(vectorizer, fp)

        if feature_type == 'abs':
            X_train, X_dev, X_test = X_train, X_dev, X_test

        if feature_type == 'rel':
            # convert to relative frequencies #TODO: Achtung, hier kann Division durch 0 auftreten, wie damit umgehen?
            X_train_rel = X_train/X_train.sum(axis=1, keepdims=True)
            X_dev_rel = X_dev/X_dev.sum(axis=1, keepdims=True)
            X_test_rel = X_test/X_test.sum(axis=1, keepdims=True)

            np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_rel_{str(max_features)}.npy', X_train_rel)
            np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_rel_{str(max_features)}.npy', X_dev_rel)
            np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_rel_{str(max_features)}.npy', X_test_rel)

            X_train, X_dev, X_test = X_train_rel, X_dev_rel, X_test_rel

        if feature_type == 'bool':
            # convert to binary values
            X_train_bool = X_train > 0
            X_dev_bool = X_dev > 0
            X_test_bool = X_test > 0

            np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_bool_{str(max_features)}.npy', X_train_bool)
            np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_bool_{str(max_features)}.npy', X_dev_bool)
            np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_bool_{str(max_features)}.npy', X_test_bool)

            X_train, X_dev, X_test = X_train_bool, X_dev_bool, X_test_bool

    if create_or_load == 'load':
        print("loading features...", feature_type)
        X_train = np.load(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_{feature_type}_{str(max_features)}.npy')
        X_dev = np.load(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_{feature_type}_{str(max_features)}.npy')
        X_test = np.load(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_{feature_type}_{str(max_features)}.npy')

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
    print("RAE:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [ p.round(2) for p in y_dev[:10] ])
    print("pred:", [ p.round(2) for p in pred_dev[:10] ])

    # saving model with pickle
    target_path = utils.OUTPUT / 'saved_models' / f'BOW_{feature_type}_{str(max_features)}.pkl'
    pickle.dump(model, open(target_path, 'wb'))



# get data
TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'


# preprocessor
preprocessor = utils.Preprocessor(delete_stopwords=True, lemmatize=True, delete_punctuation=False)

train_BOW_model(publisher = PUBLISHER,
                preprocessor = preprocessor,
                feature_type = 'abs',
                create_or_load ='create',
                max_features = 1000, #500, #TODO: warum ist hier so wenig deutlich besser als zB 1000?
                text_base = 'article_text', #'teaser', #'text_preprocessed',
                target = TARGET
                )


# to load the model
# model = pickle.load(open(target_path, 'rb'))


# bei max_features = 300, Pearson 0.666, MAE 0.187, MSE 0.072

# neue Daten (groß, aber nur SZ und TV), modelling avgTimeOnPage
# bei 100: r: 0.214, MSE: 1613.294, MAE: 29.080
# bei 500: r: 0.295, MSE: 1544.790, MAE:  28.463
# bei 1000 : r: 0.319, MSE: 1523.388, MAE: 28.100
# bei 2000: r: 0.343, MSE: 1397.636, 27.701
# bei 5000: r: 0.323, MSE: 1581.458, MAE: 28.897

# neuere Daten (mit NOZ dabei)
# bei 100: r: 0.541, MSE: 9148.856, MAE: 57.0092
# bei 500: r: 0.577, MSE: 9284.085, MAE: 54.655

# nur innerhalb NOZ:
#500 Pearson: 0.469, MSE: 19189.622, MAE: 69.543

# aktuell (NOZ):
# 500: Pearson:  0.47, MSE:  15171.37, MAE:  65.71, RAE: 92.201
# 200: Pearson:  0.45, MSE:  15498.22, MAE:  65.13, RAE: 91.386
# 1000: Pearson:  0.48, MSE:  15163.24, MAE:  66.56, RAE: 93.392