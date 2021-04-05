#!/usr/bin/env python

import numpy as np
from master_thesis.src import utils, data
from sklearn.linear_model import Ridge
import pickle
import scipy.stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import pandas as pd


def train_Embs_model(preprocessor,
                     embs,
                     publisher = 'NOZ',
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

    # get features (averaged vector of all tokens in text)
    print("creating features, so: averaging vectors")
    X_train = np.array([utils.get_averaged_vector(text, preprocessor=preprocessor, embs=embs) for text in df_train[text_base]])
    X_dev = np.array([utils.get_averaged_vector(text, preprocessor=preprocessor, embs=embs) for text in df_dev[text_base]])
    X_test = np.array([utils.get_averaged_vector(text, preprocessor=preprocessor, embs=embs) for text in df_test[text_base]])

    print("Feature shapes: ", X_train.shape, X_dev.shape, X_test.shape)

    np.save(utils.OUTPUT / 'Embs_features' / f'X_train.npy', X_train)
    np.save(utils.OUTPUT / 'Embs_features' / f'X_dev.npy', X_dev)
    np.save(utils.OUTPUT / 'Embs_features' / f'X_test.npy', X_test)

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
    #pred_dev[pred_dev < 0] = 0

    # Pearson's r and MSE as evaluation metric
    print("text_base:", text_base)
    print("target:", target)
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))
    print("MSE: ", mean_squared_error(pred_dev, y_dev))
    print("MAE: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [p.round(2) for p in y_dev[:10]])
    print("pred:", [p.round(2) for p in pred_dev[:10]])

    # saving model with pickle
    target_path = utils.OUTPUT / 'saved_models' / 'Embs.pkl'
    pickle.dump(model, open(target_path, 'wb'))




if __name__ == "__main__":

    embs = utils.load_fasttext_vectors(limit = None)
    preprocessor = utils.Preprocessor(lemmatize=False,
                                      delete_stopwords=True,
                                      delete_punctuation=True)

    example = utils.get_averaged_vector("Haus",
                                       preprocessor = preprocessor,
                                       embs = embs)
    example2 = utils.get_averaged_vector("Häuser",
                                        preprocessor=preprocessor,
                                        embs=embs)
    example3 = utils.get_averaged_vector("Häuser und",
                                         preprocessor=preprocessor,
                                         embs=embs)
    print(example[:5])
    print(example2[:5])
    print(example3[:5])


    train_Embs_model(publisher = "NOZ",
                     preprocessor = preprocessor,
                     embs = embs,
                     text_base = 'article_text',
                     target = 'avgTimeOnPage')


    #PUBLISHER = "NOZ"
    #full = utils.get_publisher_df(PUBLISHER)

    #X = np.array([utils.get_averaged_vector(text, preprocessor=preprocessor, embs=embs) for text in full["article_text"]])

    #X_fastText = pd.DataFrame(data = X, index = full.index, columns = range(300)) # for saving document embeddings
    #print(X_fastText.head())
    #print(X_fastText.shape)

    #X_fastText.to_csv(utils.OUTPUT / 'Embs_features' / f'Embs_features_{PUBLISHER}_full_nonlemmatized.tsv',
    #             sep ='\t', index_label="articleId")
    


# NOZ (aktuell)

# lemmatize=False, delete_stopwords=True, delete_punctuation=True)
# Pearson: 0.39
# MSE: 16551.1
# MAE: 67.8
# RAE: 95.1

# lemmatize=True, delete_stopwords=True, delete_punctuation=True)
# Pearson: 0.39
# MSE: 17152.6
# MAE: 68.6
# RAE: 96.3