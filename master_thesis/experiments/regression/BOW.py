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
        X_train = np.load(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_X_train_abs_{str(max_features)}_{preprocessing}.npy')
        X_dev = np.load(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_X_dev_abs_{str(max_features)}_{preprocessing}.npy')
        X_test = np.load(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_X_test_abs_{str(max_features)}_{preprocessing}.npy')

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

        if vectorizer_type == "TfidfVectorizer":
            print("Using TfidfVectorizer")
            vectorizer = TfidfVectorizer(analyzer='word',
                                         preprocessor=preprocessor,
                                         ngram_range=(1, MAX_NGRAM),
                                         min_df=MIN_DF,  # Achtung: vielleicht je nach Textbasis anpassen?
                                         max_features=max_features
                                         )
        print("fitting vectorizer and transforming texts to features...")
        X_train = vectorizer.fit_transform(df_train[text_base])
        X_dev = vectorizer.transform(df_dev[text_base])
        X_test = vectorizer.transform(df_test[text_base])

        X_train, X_dev, X_test = X_train.toarray(), X_dev.toarray(), X_test.toarray()

        np.save(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_X_train_abs_{str(max_features)}_{preprocessing}.npy', X_train)
        np.save(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_X_dev_abs_{str(max_features)}_{preprocessing}.npy', X_dev)
        np.save(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_X_test_abs_{str(max_features)}_{preprocessing}.npy', X_test)

        # saving vectorizer (for inference and for having feature_names)
        with open(utils.OUTPUT / 'BOW_features' / f'{vectorizer_type}_{publisher}_vectorizer_{str(max_features)}_{preprocessing}.pkl', "wb") as fp:
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

        X_train, X_dev, X_test = X_train_rel, X_dev_rel, X_test_rel

    if feature_type == 'bin':
        print("using binary frequency (present or not)")

        # convert to binary values
        X_train_bool = X_train > 0
        X_dev_bool = X_dev > 0
        X_test_bool = X_test > 0

        X_train, X_dev, X_test = X_train_bool, X_dev_bool, X_test_bool


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
        model = xgboost.XGBRegressor(n_estimators = 300, #300,
                                     learning_rate = 0.1,
                                     max_depth = 10, #10,
                                     random_state = 1,
                                     subsample= 1, # 1# was ist das genau? "fraction of observations to be randomly samples for each tree"
                                     #tree_method='gpu_hist', # doesn't work
                                     #gpu_id=0
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
    df_dev.to_csv(utils.OUTPUT / "predictions" / "dev" / f'BOW_{vectorizer_type}_{classifier}_{feature_type}_{str(max_features)}_{preprocessing}.tsv', sep="\t", index=True,
                  index_label="articleId")


    print("predicting test set")
    pred_test = model.predict(X_test)

    df_test = pd.DataFrame(0., index=test_IDs, columns=["true", "pred"])
    df_test["true"] = y_test
    df_test["pred"] = pred_test
    print(df_test.head())
    df_test.to_csv(utils.OUTPUT / "predictions" / "test" / f'BOW_{vectorizer_type}_{classifier}_{feature_type}_{str(max_features)}_{preprocessing}.tsv', sep="\t", index=True,
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
    target_path = utils.OUTPUT / 'saved_models' / f'BOW_{vectorizer_type}_{classifier}_{feature_type}_{str(max_features)}_{preprocessing}.pkl'
    pickle.dump(model, open(target_path, 'wb'))



# get data
TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'


# preprocessor
preprocessor = utils.Preprocessor(delete_stopwords=False, lemmatize=False, delete_punctuation=True)
preprocessing = f'{preprocessor.delete_stopwords}{preprocessor.lemmatize}{preprocessor.delete_punctuation}'
print(preprocessing)

train_BOW_model(classifier= "ridge", #"ridge", #"xgboost",
                vectorizer_type= "CountVectorizer", #"CountVectorizer", # "TfidfVectorizer"
                publisher = PUBLISHER,
                preprocessor = preprocessor,
                feature_type = 'abs',
                load = True,
                max_features = 50000, #500, #TODO: warum ist hier so wenig deutlich besser als zB 1000?
                text_base = 'article_text', #'teaser', #'text_preprocessed',
                target = TARGET
                )


# to load the model
# model = pickle.load(open(target_path, 'rb'))


# Count Vectorizer + ridge
# 500: Pearson:  0.47, MSE:  15171.37, MAE:  65.71, RAE: 92.201
# 200: Pearson:  0.45, MSE:  15498.22, MAE:  65.13, RAE: 91.386
# 1000: Pearson:  0.48, MSE:  15163.24, MAE:  66.56, RAE: 93.392
# 1000 (true,true,true) 0.47
# 1000 true true true Pearson: 0.474 Spearman: 0.42  MSE:  15197.4381 MAE:  66.757   RAE: 93.667      ***** current best
# 5000 true true true Pearson: 0.457 Spearman 0.37   MSE:  16468.7639 MAE:  77.114  RAE: 108.1987
# 10000 rel 0.37, abs: 0.40, bin: 0.40
# 10000 (true,true,true) 0.40
# 10000 true true true Pearson: 0.39 Spearman: 0.309 MSE:  20569.1089 MAE:  92.066   RAE: 129.177
# 10000 false false true ... Pearson:  0.39 Spearman: 0.329 MSE:  20589.844  MAE:  91.566 RAE: 128.47
# 50000 false true true Pearson 0.29 Spearman 0.232  MSE:  32424.187  MAE:  124.643  RAE: 174.886
# 50000 true true true Pearson: 0.315 Spearman: 0.268 MSE:  30769.86 MAE:  119.180 RAE: 167.220
# 50000 false false true ... Pearson: 0.2885
# 100000 false true true Pearson: 0.379 Spearman: 0.323 MSE:  22134.239 MAE:  95.468 RAE: 133.9



# CountVectorizer + xgboost
# 500: Pearson: 0.48
# 1000: Pearson: 0.47 (binary: 0.38)
# 10000: Pearson: 0.49 (binary: 0.47)
# 10000 (false true true) 0.51, 60, 84, 14450
# 50000 (false true true) 0.50, 60, 84

# 1000 true true true  XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#       Pearson: 0.461 Spearman: 0.5351 MSE:  15486.54 MAE:  62.205 RAE: 87.27
# 5000 true true true XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#       Pearson: 0.481 Spearman: 0.568 MSE:  15027.175 MAE:  60.627 RAE: 85.066
# 10000 false true true, XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#       Pearson 0.525 Spearman: 0.562 MSE:  14129.26 MAE:  59.51 RAE: 83.506
# 10000 false true true XGBRegressor(max_depth=50, random_state=1)
#      Pearson 0.471  Spearman:  0.5194  MSE:  15206.427  MAE:  62.5596  RAE: 87.776
# 10000 false true true XGBRegressor(max_depth=20, n_estimators=500, random_state=1)
#      Pearson: 0.509  Spearman:  0.562 MSE:  14451.583 MAE:  60.00  RAE: 84.190
# 10000 false true true XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#      Pearson: 0.525 Spearman: 0.562 MSE:  14129.268 MAE:  59.516 RAE: 83.506
# 10000 true true true XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#      Pearson:  0.499 Spearman: 0.557 MSE:  14657.376 MAE:  60.81 RAE: 85.32
# 10000 false false true ... max_dept 10 n_estimators 300
#      Pearson: 0.547 Spearman: 0.565  MSE:  13687.544  MAE:  59.080 RAE: 82.894       ******* current best

# 50000 false true true XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#      Pearson:  0.531  Spearman:  0.569  MSE:  14014.80  MAE:  58.82  RAE: 82.53
# 50000 false true true XBGRegressor(max_depth=5, n_estimators 100, random_state=1)
#      Pearson: 0.506 Spearman: 0.543 MSE:  14596.383 MAE:  59.416 RAE: 83.366
# 50000 true true true XGBRegressor(max_depth=10, n_estimators=300, random_state=1)
#      Pearson: 0.484  Spearman: 0.553  MSE:  14965.402  MAE:  60.390  RAE: 84.733
# 50000 false false true ... (10, 300)
#      Pearson:  (0.503 Spearman: 0.559  MSE:  14582.689  MAE:  59.9144 RAE: 84.065
# 100000 false true true (max_depth = 5, n_estimators = 100)
#        Pearson: 0.504 Spearmanr 0.545 MSE:  14662.452 MAE:  59.361 RAE: 83.289
# 100000 false true true (max depth = 10, n_estimators = 300
#        Pearson: 0.528 Spearman 0.569 MSE:  14092.393 MAE:  58.555  RAE: 82.157
# 100000 false true true (max_depth 20, n_estimators = 300
#        Pearson: 0.501 Spearman 0.568 MSE:  14635.104  MAE:  59.772 RAE: 83.866
# 100000 false true true (max_depth = 10, n_estimators = 800)
#        Pearson: 0.530 Spearman 0.573  MSE:  14035.741 MAE:  58.700 RAE: 82.362


# tfidf + ridge
# 500: 0.41
# 1000: 0.44
# 10000: 0.47 (false true true)
# 10000: 0.48  (true true true)

# tfidf + xgboost
# 500 (false true true) 0.46
# 1000 (false true true) 0.45
