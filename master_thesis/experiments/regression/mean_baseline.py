#!/usr/bin/env python

from master_thesis.src import utils, data

import numpy as np
import scipy.stats as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json


def mean_baseline(publisher, target):
    print("Using mean as baseline...")
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

    # define labels
    y_dev = np.array(df_dev[target])
    y_test = np.array(df_test[target])

    mean_train = np.mean(df_train[target])

    # predict for dev set
    print("predicting dev set")
    pred_dev = np.full(len(y_dev), mean_train)

    # Pearson's r and MSE as evaluation metric
    print("target:", target)
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))
    print("MSE: ", mean_squared_error(pred_dev, y_dev))
    print("MAE: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [p.round(2) for p in y_dev[:10]])
    print("pred:", [p.round(2) for p in pred_dev[:10]])


def textlength_baseline(publisher, target):
    print("Using regression with just textlength as feature...")
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

    # features
    X_train = np.array(df_train['nr_tokens_text_BERT'])   #     X_train = np.array(df_train['nr_tokens_text'])
    X_dev = np.array(df_dev['nr_tokens_text_BERT'])       #     X_dev = np.array(df_dev['nr_tokens_text'])
    X_test = np.array(df_test['nr_tokens_text_BERT'])     #     X_test = np.array(df_test['nr_tokens_text'])

    # define labels
    y_train = np.array(df_train[target])
    y_dev = np.array(df_dev[target])
    y_test = np.array(df_test[target])

    print("Labels shape: ", y_train.shape, y_dev.shape, y_test.shape)

    # model: Ridge Regression
    model = Ridge()

    print("training model...")
    model.fit(X_train.reshape(-1, 1), y_train)

    # predict for dev set
    print("predicting dev set")
    pred_dev = model.predict(X_dev.reshape(-1, 1))

    # Pearson's r and MSE as evaluation metric
    print("target:", target)
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))
    print("MSE: ", mean_squared_error(pred_dev, y_dev))
    print("MAE: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [ p.round(2) for p in y_dev[:10] ])
    print("pred:", [ p.round(2) for p in pred_dev[:10] ])



# get data
TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'


mean_baseline(publisher = PUBLISHER,
              target = TARGET
              )

print("--------------------------------------")

textlength_baseline(publisher = PUBLISHER,
                    target= TARGET)


# NOZ:

### mean_baseline:
# Pearson:  (nan, nan) (not defined because pred is a constant)
# MSE:  19533.40
# MAE:  70.21
# RAE: 98.51

### textlength_baseline:
## with "normal" textlength:
# Pearson:  0.39
# MSE:  17832.166
# MAE:  63.931
# RAE: 89.702

## with BERT_tokens:
# Pearson: 0.38
# MSE: 18057.853
# MAE: 64.635
# RAE: 90.689