#!/usr/bin/env python

from master_thesis.src import utils, data

import numpy as np
import pandas as pd
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

    # predict for dev/test set
    print("predicting dev/test set")
    pred_dev = np.full(len(y_dev), mean_train)
    df_dev = pd.DataFrame(0., index=df_dev.index, columns=["true", "pred"])
    df_dev["true"] = y_dev
    df_dev["pred"] = pred_dev
    df_dev.to_csv(utils.OUTPUT / "predictions" / "dev" / f'mean_baseline.tsv', sep="\t", index=True,
                  index_label="articleId")

    pred_test = np.full(len(y_test), mean_train)
    df_test = pd.DataFrame(0., index=df_test.index, columns=["true", "pred"])
    df_test["true"] = y_test
    df_test["pred"] = pred_test
    df_test.to_csv(utils.OUTPUT / "predictions" / "test" / f'mean_baseline.tsv', sep="\t", index=True,
                   index_label="articleId")

    # Pearson's r and MSE as evaluation metric
    print("target:", target)
    print("Pearson DEV: ", st.pearsonr(pred_dev, y_dev))
    print("Spearman DEV: ", st.spearmanr(pred_dev, y_dev)[0])
    print("MSE DEV: ", mean_squared_error(pred_dev, y_dev))
    print("MAE DEV: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE DEV:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [p.round(2) for p in y_dev[:10]])
    print("pred:", [p.round(2) for p in pred_dev[:10]])

    # Pearson's r and MSE as evaluation metric
    print("target:", target)
    print("Pearson TEST: ", st.pearsonr(pred_test, y_test))
    print("Spearman TEST: ", st.spearmanr(pred_test, y_test)[0])
    print("MSE TEST: ", mean_squared_error(pred_test, y_test))
    print("MAE TEST: ", mean_absolute_error(pred_test, y_test))
    print("RAE TEST:", utils.relative_absolute_error(pred_test, y_test))

    # print some predictions
    print("true:", [p.round(2) for p in y_test[:10]])
    print("pred:", [p.round(2) for p in pred_test[:10]])


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
    X_train = np.array(df_train['nr_tokens_text'])
    X_dev = np.array(df_dev['nr_tokens_text'])
    X_test = np.array(df_test['nr_tokens_text'])

    #X_train = np.array(df_train['nr_tokens_text_BERT'])
    #X_dev = np.array(df_dev['nr_tokens_text_BERT'])
    #X_test = np.array(df_test['nr_tokens_text_BERT'])


    # define labels
    y_train = np.array(df_train[target])
    y_dev = np.array(df_dev[target])
    y_test = np.array(df_test[target])

    print("Labels shape: ", y_train.shape, y_dev.shape, y_test.shape)

    # model: Ridge Regression
    model = Ridge()

    print("training model...")
    model.fit(X_train.reshape(-1, 1), y_train)

    # predict for dev/test set
    print("predicting dev set")
    pred_dev = model.predict(X_dev.reshape(-1, 1))
    df_dev = pd.DataFrame(0., index=df_dev.index, columns=["true", "pred"])
    df_dev["true"] = y_dev
    df_dev["pred"] = pred_dev
    df_dev.to_csv(utils.OUTPUT / "predictions" / "dev" / f'textlength_baseline.tsv', sep="\t", index=True,
                  index_label="articleId")

    print("predicting test set")
    pred_test = model.predict(X_test.reshape(-1, 1))
    df_test = pd.DataFrame(0., index=df_test.index, columns=["true", "pred"])
    df_test["true"] = y_test
    df_test["pred"] = pred_test
    df_test.to_csv(utils.OUTPUT / "predictions" / "test" / f'textlength_baseline.tsv', sep="\t", index=True,
                  index_label="articleId")

    # Pearson's r and MSE as evaluation metric
    print("target:", target)
    print("Pearson DEV: ", st.pearsonr(pred_dev, y_dev))
    print("Spearman DEV: ", st.spearmanr(pred_dev, y_dev)[0])
    print("MSE DEV: ", mean_squared_error(pred_dev, y_dev))
    print("MAE DEV: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE DEV:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [ p.round(2) for p in y_dev[:10] ])
    print("pred:", [ p.round(2) for p in pred_dev[:10] ])

    # Pearson's r and MSE as evaluation metric
    print("target:", target)
    print("Pearson TEST: ", st.pearsonr(pred_test, y_test))
    print("Spearman TEST: ", st.spearmanr(pred_test, y_test)[0])
    print("MSE TEST: ", mean_squared_error(pred_test, y_test))
    print("MAE TEST: ", mean_absolute_error(pred_test, y_test))
    print("RAE TEST:", utils.relative_absolute_error(pred_test, y_test))

    # print some predictions
    print("true:", [ p.round(2) for p in y_test[:10] ])
    print("pred:", [ p.round(2) for p in pred_test[:10] ])


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