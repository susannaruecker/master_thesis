#!/usr/bin/env python

import numpy as np
from master_thesis.src import utils, data
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor
import pickle
import scipy.stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from master_thesis.src import utils, data, models

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("device", help="specify which device to use ('cpu' or 'gpu')", type=str)
args = parser.parse_args()


def create_features(FIXED_LEN = 128):

    device = torch.device('cpu' if args.device == 'cpu' else 'cuda')
    print('Using device:', device)

    # get pretrained model and tokenizer from huggingface's transformer library
    PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    model = models.BERT_embedding()
    model.to(device)

    PUBLISHER = "NOZ"
    START = 0
    FIXED_LEN = FIXED_LEN
    MIN_LEN = None
    FRACTION = 1
    TARGET = "avgTimeOnPage"
    GPU_BATCH = 100 #500

    transform = data.TransformBERT(tokenizer = tokenizer, start = START, fixed_len = FIXED_LEN, min_len= MIN_LEN)
    collater = data.CollaterBERT()

    ds_train = data.PublisherDataset(publisher=PUBLISHER, set = "train", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
    ds_dev = data.PublisherDataset(publisher=PUBLISHER, set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
    ds_test = data.PublisherDataset(publisher=PUBLISHER, set = "test", fraction=FRACTION, target  = TARGET, text_base = "article_text", transform = transform)
    print("Length of used DataSets:", len(ds_train), len(ds_dev), len(ds_test))

    dl_train = DataLoader(ds_train, batch_size=GPU_BATCH, num_workers=0, shuffle=False, collate_fn=collater, drop_last = False)
    dl_dev = DataLoader(ds_dev, batch_size=GPU_BATCH, num_workers=0, shuffle=False, collate_fn=collater, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=GPU_BATCH, num_workers=0, shuffle=False, collate_fn=collater, drop_last=False)

    # getting the IDs for the sets
    with open(utils.OUTPUT / "splits" / f"{PUBLISHER}_splits.json", "r") as f:
        splits = json.load(f)
        train_IDs = splits["train"]
        dev_IDs = splits["dev"]
        test_IDs = splits["test"]

    X_train = pd.DataFrame(index = train_IDs, columns = range(768))
    X_dev = pd.DataFrame(index = dev_IDs, columns = range(768))
    X_test = pd.DataFrame(index = test_IDs, columns = range(768))


    # running the "model" (just getting BERT-Embeddings, so no backpropagation/optimizer etc.)
    with torch.no_grad():
        for nr, d in enumerate(dl_train):
            print("nr:", nr, end='\r')
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            articleId = d['articleId']
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.cpu()
            for i, emb in enumerate(embedding):
                X_train.loc[articleId[i]] = emb
            #if nr >= 3:
            #    break

        print("done with train, saving X_train")
        X_train.to_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_train_FIXLEN{FIXED_LEN}.tsv',
                       sep = '\t', index_label = "articleId")

        for nr, d in enumerate(dl_dev):
            print("nr:", nr, end='\r')
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            articleId = d['articleId']
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            for i, emb in enumerate(embedding):
                X_dev.loc[articleId[i]] = emb.cpu()
        print("done with dev, saving X_dev")
        X_dev.to_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_dev_FIXLEN{FIXED_LEN}.tsv',
                       sep='\t', index_label="articleId")

        for nr, d in enumerate(dl_test):
            print("nr:", nr, end='\r')
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            articleId = d['articleId']
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            for i, emb in enumerate(embedding):
                X_test.loc[articleId[i]] = emb.cpu()
        print("done with test, saving X_test")
        X_test.to_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_test_FIXLEN{FIXED_LEN}.tsv',
                       sep='\t', index_label="articleId")


if __name__ == "__main__":

    # only necessary one time for the fixed len...
    FIXED_LEN = 512
    create_features(FIXED_LEN = 512)

    # load features
    PUBLISHER = "NOZ"
    X_train_df = pd.read_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_train_FIXLEN{FIXED_LEN}.tsv',
                       sep='\t', index_col="articleId")
    X_dev_df = pd.read_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_dev_FIXLEN{FIXED_LEN}.tsv',
                       sep='\t', index_col="articleId")
    X_test_df = pd.read_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_test_FIXLEN{FIXED_LEN}.tsv',
                       sep='\t', index_col="articleId")

    # concatenate and save them
    X_full_df = pd.concat([X_train_df, X_dev_df, X_test_df])
    X_full_df.to_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_{PUBLISHER}_full_FIXLEN{FIXED_LEN}.tsv',
                 sep='\t', index_label="articleId")

    # convert to numpy
    X_train, X_dev, X_test = np.array(X_train_df), np.array(X_dev_df), np.array(X_test_df)
    print("Features shape:", X_train.shape, X_dev.shape, X_test.shape)

    # get true labels
    df = utils.get_publisher_df(PUBLISHER)
    df_train = df.loc[X_train_df.index]
    df_dev = df.loc[X_dev_df.index]
    df_test = df.loc[X_test_df.index]

    TARGET = "avgTimeOnPage"
    y_train = np.array(df_train[TARGET])
    y_dev = np.array(df_dev[TARGET])
    y_test = np.array(df_test[TARGET])

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
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))
    print("MSE: ", mean_squared_error(pred_dev, y_dev))
    print("MAE: ", mean_absolute_error(pred_dev, y_dev))
    print("RAE:", utils.relative_absolute_error(pred_dev, y_dev))

    # print some predictions
    print("true:", [p.round(2) for p in y_dev[:10]])
    print("pred:", [p.round(2) for p in pred_dev[:10]])

    # saving model with pickle
    target_path = utils.OUTPUT / 'saved_models' / f'BERT_features_FIXLEN{FIXED_LEN}.pkl'
    pickle.dump(model, open(target_path, 'wb'))

# NOZ (erste 128 Tokens)
# Pearson:  (0.4271697683726695, 2.62008033037408e-161)
# MSE:  16001.888120996331
# MAE:  70.85979305771858
# RAE: 99.42258413583295

# NOZ (erste 512 Tokens)
# TODO
# Pearson: 0.514
# MSE: 14379
# MAE: 66.5
# RAE: 93.3


