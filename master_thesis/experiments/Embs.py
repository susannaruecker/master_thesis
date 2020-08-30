import numpy as np
from master_thesis.src import utils, data
from sklearn.linear_model import Ridge
import pickle
import scipy.stats as st
import pandas as pd


def train_Embs_model(df,
                     preprocessor,
                     embs,
                     text_base = 'text_preprocessed',
                     target = 'avgTimeOnPagePerNr_tokens'
                     ):
    # create splits
    df_train, df_dev, df_test = data.create_train_dev_test(df=df, random_seed=123)
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
    pred_dev[pred_dev < 0] = 0

    # Pearson's r as evaluation metric
    print("Pearson: ", st.pearsonr(pred_dev, y_dev))

    # saving model with pickle
    target_path = utils.OUTPUT / 'saved_models' / 'Embs.pkl'
    pickle.dump(model, open(target_path, 'wb'))

embs = utils.load_fasttext_vectors(limit = None)

preprocessor = utils.Preprocessor(lemmatize=True,
                                  delete_stopwords=True,
                                  delete_punctuation=True)

#example = utils.get_averaged_vector("Haus und Hof.",
#                                   preprocessor = preprocessor,
#                                   embs = embs)


# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()

train_Embs_model(df = df,
                 preprocessor = preprocessor,
                 embs = embs,
                 text_base = 'text_preprocessed',
                 target = 'avgTimeOnPagePerNr_tokens')

# note: hier kam raus: Pearson: (0.3800609554415717, 3.8111553582181078e-149) # ohne Lemmatisierung, mit Stopwortentfernung und mit Puncutation-Entfernung
