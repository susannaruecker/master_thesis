#!/usr/bin/env python

from master_thesis.src import utils, data

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import Ridge
import xgboost
import scipy.stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
import matplotlib.pyplot as plt


def train_BOW_model(classifier = "ridge",
                    vectorizer_type = "CountVectorizer",
                    publisher = 'NOZ',
                    preprocessor = None,
                    feature_type = 'abs',
                    load = True,
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

    if load == True:
        print("loading features...")
        X_train = np.load(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_abs_{str(max_features)}.npy')
        X_dev = np.load(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_abs_{str(max_features)}.npy')
        X_test = np.load(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_abs_{str(max_features)}.npy')

    else:
        print("creating features...")
        print("max_features = ", max_features)


        # instantiate Vectorizer
        MAX_NGRAM = 4
        MIN_DF = 5

        if vectorizer_type == "CountVectorizer":
            print("Using CountVectorizer")
            vectorizer = CountVectorizer(analyzer='word',
                                         preprocessor= preprocessor,
                                         ngram_range= (1, MAX_NGRAM),
                                         min_df = MIN_DF, # Achtung: vielleicht je nach Textbasis anpassen?
                                         max_features = max_features
                                         )

            print("fitting vectorizer and transforming texts to features...")
            X_train = vectorizer.fit_transform(df_train[text_base])
            X_dev = vectorizer.transform(df_dev[text_base])
            X_test = vectorizer.transform(df_test[text_base])

            X_train, X_dev, X_test = X_train.toarray(), X_dev.toarray(), X_test.toarray()

            #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_abs_{str(max_features)}.npy', X_train)
            #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_abs_{str(max_features)}.npy', X_dev)
            #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_abs_{str(max_features)}.npy', X_test)

            # saving vectorizer (for inference and for having feature_names)
            with open(utils.OUTPUT / 'BOW_features' / f'{publisher}_vectorizer_{str(max_features)}.pkl', "wb") as fp:
                pickle.dump(vectorizer, fp)

            if feature_type == 'abs':
                print("using absolute frequency")
                X_train, X_dev, X_test = X_train, X_dev, X_test

            if feature_type == 'rel':
                print("using relative frequency")
                # convert to relative frequencies #TODO: Achtung, hier kann Division durch 0 auftreten, wie damit umgehen?
                X_train_rel = X_train/X_train.sum(axis=1, keepdims=True)
                X_dev_rel = X_dev/X_dev.sum(axis=1, keepdims=True)
                X_test_rel = X_test/X_test.sum(axis=1, keepdims=True)

                #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_rel_{str(max_features)}.npy', X_train_rel)
                #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_rel_{str(max_features)}.npy', X_dev_rel)
                #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_rel_{str(max_features)}.npy', X_test_rel)

                X_train, X_dev, X_test = X_train_rel, X_dev_rel, X_test_rel

            if feature_type == 'bin':
                print("using binary frequency (present or not)")

                # convert to binary values
                X_train_bool = X_train > 0
                X_dev_bool = X_dev > 0
                X_test_bool = X_test > 0

                #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_train_bool_{str(max_features)}.npy', X_train_bool)
                #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_dev_bool_{str(max_features)}.npy', X_dev_bool)
                #np.save(utils.OUTPUT / 'BOW_features' / f'{publisher}_X_test_bool_{str(max_features)}.npy', X_test_bool)

                X_train, X_dev, X_test = X_train_bool, X_dev_bool, X_test_bool

        if vectorizer_type == "TfidfVectorizer":
            print("Using TfidfVectorizer, no saving...")
            vectorizer = TfidfVectorizer(analyzer='word',
                                         preprocessor=preprocessor,
                                         ngram_range=(1, MAX_NGRAM),
                                         min_df=MIN_DF,  # Achtung: vielleicht je nach Textbasis anpassen?
                                         max_features=max_features
                                         )
            X_train = vectorizer.fit_transform(df_train[text_base])
            X_dev = vectorizer.transform(df_dev[text_base])
            X_test = vectorizer.transform(df_test[text_base])

            X_train, X_dev, X_test = X_train.toarray(), X_dev.toarray(), X_test.toarray()

    print("Feature shapes: ", X_train.shape, X_dev.shape, X_test.shape)
    print(X_train[:3])

    # define labels
    y_train = np.array(df_train[target])
    y_dev = np.array(df_dev[target])
    y_test = np.array(df_test[target])

    print("Labels shape: ", y_train.shape, y_dev.shape, y_test.shape)

    if classifier == "ridge":
        model = Ridge()

    if classifier == "xgboost":
        model = xgboost.XGBRegressor(n_estimators = 800, #800, #800
                                     learning_rate = 0.1,
                                     max_depth = 20, # 20
                                     #verbosity=2
                                     random_state = 1,
                                     subsample=1 # was ist das genau? "fraction of observations to be randomly samples for each tree"
                                     )

    print("using model:", model)

    print("training model...")
    model.fit(X_train, y_train)
    print("fitted model:", model)
    # predict for dev set
    print("predicting dev set")
    pred_dev = model.predict(X_dev)

    df_dev = pd.DataFrame(0., index=dev_IDs, columns=["true", "pred"])
    df_dev["true"] = y_dev
    df_dev["pred"] = pred_dev
    print(df_dev.head())
    df_dev.to_csv(utils.OUTPUT / "predictions" / "dev" / f'BOW_{classifier}_{feature_type}_{str(max_features)}.tsv', sep="\t", index=True,
                  index_label="articleId")


    print("predicting test set")
    pred_test = model.predict(X_test)

    df_test = pd.DataFrame(0., index=test_IDs, columns=["true", "pred"])
    df_test["true"] = y_test
    df_test["pred"] = pred_test
    print(df_test.head())
    df_test.to_csv(utils.OUTPUT / "predictions" / "test" / f'BOW_{classifier}_{feature_type}_{str(max_features)}.tsv', sep="\t", index=True,
                  index_label="articleId")

    #if target == "log_avgTimeOnPage":
    #    "transforming log values back to avgTimeOnPage"
    #    pred_dev = np.exp(pred_dev)
    #    y_dev = np.exp(y_dev)

    # Pearson's r and MSE as evaluation metric
    print("text_base:", text_base)
    print("target:", target)
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))
    print("Spearman: ", st.spearmanr(pred_dev, y_dev))
    print("MSE: ", mean_squared_error(pred_dev, y_dev))
    print("MAE: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [ p.round(2) for p in y_dev[:10] ])
    print("pred:", [ p.round(2) for p in pred_dev[:10] ])

    # saving model with pickle
    #target_path = utils.OUTPUT / 'saved_models' / f'BOW_{classifier}_{feature_type}_{str(max_features)}.pkl'
    #pickle.dump(model, open(target_path, 'wb'))



# get data
TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'


# preprocessor
preprocessor = utils.Preprocessor(delete_stopwords=False, lemmatize=True, delete_punctuation=True)
print(preprocessor.delete_stopwords, preprocessor.lemmatize, preprocessor.delete_punctuation)

train_BOW_model(classifier= "ridge", #"ridge", #"xgboost",
                vectorizer_type= "CountVectorizer", #"CountVectorizer", # "TfidfVectorizer"
                publisher = PUBLISHER,
                preprocessor = preprocessor,
                feature_type = 'abs',
                load = False,
                max_features = 10000, #500, #TODO: warum ist hier so wenig deutlich besser als zB 1000?
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
# 10000 rel 0.37, abs: 0.40, bin: 0.40
# 10000 (true,true,true) 0.40
# 1000 (true,true,true) 0.47

# CountVectorizer + xgboost
# 500: Pearson: 0.48
# 1000: Pearson: 0.47 (binary: 0.38)
# 10000: Pearson: 0.49 (binary: 0.47)
# 10000 (false true true) 0.51, 60, 84, 14450
# 50000 (false true true) 0.50, 60, 84

# tfidf + ridge
# 500: 0.41
# 1000: 0.44
# 10000: 0.47 (false true true)
# 10000: 0.48  (true true true)

# tfidf + xgboost
# 500 (false true true) 0.46
# 1000 (false true true) 0.45
