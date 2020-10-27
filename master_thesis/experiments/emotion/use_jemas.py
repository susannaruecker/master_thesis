#!/usr/bin/env python

import sys
sys.path.append('/home/ruecker/workspace')

#import os
#sys.path.append(os.getcwd()) # for slurm?

from jemas4py.jemas.src import jemas
from master_thesis.src import utils
import numpy as np
import pandas as pd
import re

text = "Das hier sind Beispieltexte, bei denen die ganz doll super tollen Emotionen bestimmt werden sollen."
print(text)

# german emotion lexicon (VA), (Vo et al., 2009), neutral values = [0., 3.], scale valence = [-3,3], scale arousal, = [1,5]
#lex = utils.get_Vo()
#neutral = np.array([0.,3.])

# german emotion lexicon (Schmidke et al., 2014), neutral values = [0., 5., 5.], scale valence [-3,3], scale arousal + dominance = [1,9]
#lex = utils.get_ANGST()
#neutral = np.array([0.,5.,5.])

# MEmoLon
lex = utils.get_MEmoLon(max_length = 5000)
lex = lex.reset_index().dropna().set_index('word') # to delete nan in index...
neutral = np.array([5.,5.,5.])

print(lex.head())
print(lex.shape)
vars = lex.columns.to_list()
print("vars", vars)

#TODO: Hier überlegeb, welcher Preprocessor, muss das Lexikon auch da durch?

#preprocessor = jemas.German_Preprocessor() # for Vo-lexicon: lemmatization, lower casing, deletion of stopwords
preprocessor = jemas.Minimal_Preprocessor('de') # for MEmoLon: deleting punct and stopwords, no lemmatization

print(preprocessor(text))

# average words which belong to same lemma and make dict
lex_dict = {}
for key in set(lex.index):
    candidate = lex.loc[key]
    if isinstance(candidate, pd.Series):
        lex_dict[key] = candidate.to_numpy()
    elif isinstance(candidate, pd.DataFrame):
        lex_dict[key] = candidate.mean(axis=0).to_numpy()
    else:
        raise ValueError()
lex = lex_dict
print(len(lex))

analyzer = jemas.Analyzer(lex = lex,
                          vars = vars,
                          neutral = neutral,
                          preprocessor = preprocessor)

df_full = utils.get_raw_df()
df_full.set_index('articleId', inplace=True)
df_full = df_full[['text_preprocessed']]

# new df with emotional variables as columns
df = df_full.copy()
for v in vars:
    df[v] = np.nan

print(df.head())
print(df.shape)

IDs = df[df[vars[0]].isnull()].index.tolist() # still "empty" IDs
print('len IDs', len(IDs))

for i, ID in enumerate(IDs):
    if i % 1000 == 0:
        print(f"at ID nr {i} ...")
    #print(ID)
    #if i > 4:
    #    break
    text = df.loc[ID, 'text_preprocessed']
    rt = analyzer(text)
    #print(rt)
    #print(rt['tokens_total'])
    #print(rt['tokens_recognized'])
    #print(rt['list_tokens_recognized'])
    recognized = round(rt['tokens_recognized'] / rt['tokens_total'], 3)
    #print(recognized)

    values = rt['doc_emotion']
    #print(values)
    for j, v in enumerate(values):
        emotion = vars[j]  # vars is list of columns in lexicon
        # print(emotion)
        df.loc[ID, emotion] = np.around(v, decimals=3)
    df.loc[ID, 'token_recognition'] = recognized

print(df.head())

df.to_csv('jemas_MEmoLon_maxlen5000.tsv', sep='\t')
#df.to_csv('jemas_vo.tsv', sep='\t')
#df.to_csv('jemas_MEmoLon.tsv', sep='\t')
