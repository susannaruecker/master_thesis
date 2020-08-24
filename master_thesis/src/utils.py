import pandas as pd
import numpy as np
import string
import nltk
nltk.download('punkt')
import json
import re
import csv
from pathlib import Path

#ROOT = Path('/Volumes/INWT/Daten_NLP') # encrypted folder!
ROOT = Path('/home/ruecker/data/Daten_INWT/') # JULIE-Server

#DATA = ROOT / 'data'
DATA = ROOT / '200820_dataNLP'
META = ROOT / 'Dokumentation_Daten.txt'

PROJECT_ROOT = Path(__file__).parents[2].resolve()
OUTPUT = PROJECT_ROOT / 'master_thesis' / 'outputs' # to save tsv files and other stuff

def read_data(file):
    raw = pd.read_csv(DATA / file, engine='python', quoting=csv.QUOTE_ALL, escapechar = '\\', index_col = 'articleId')
    #raw = pd.read_csv(DATA / file, encoding='utf-8', error_bad_lines=False, quotechar = '"', doublequote=False)
    return raw

    
def create_meta_dict():
    # Zusätzliche Information zu den Spalten 2-11 finden sich unter
    # https://ga-dev-tools.appspot.com/dimensions-metrics-explorer/?
    meta_dict = {}
    with open(META, 'r') as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #print(line.strip())
            if '- (' in line:
                line = re.sub(r'- \(\d+\): ', '', line).strip()
                #print(line)
                var_name, var_description = re.split(': | - ',line)
                meta_dict[var_name] = var_description
            line = fp.readline()
    meta_dict['publisher'] = 'Angabe zum Publisher, bei dem der Artikel erschienen ist'
    with open(OUTPUT / 'meta_dict.json', 'w') as f:
        json.dump(meta_dict, f)
    return meta_dict
    

def get_meta_dict():
    with open(OUTPUT / 'meta_dict.json', 'r') as f:
        meta_dict = json.load(f)
    return meta_dict


def get_meta_cat_file(meta_cat):
    tmp_meta = pd.read_csv(OUTPUT / f'meta_file_{meta_cat}.csv', index_col=0)
    return tmp_meta


def add_meta_columns(df):
    
    df = df.fillna('') # replacing Nan with emtpy string
      
    # add columns 'city' and 'text_preprocessed'
    print("preprocessing text...")
    for n, ID in enumerate(df.index):
        print(n, end='\r')
        raw_text = df.loc[ID, 'text']
        raw_text = raw_text.replace('\xa0', ' ')
        if "(dpa" in raw_text:
            split = re.split(r'\(dpa\S*', raw_text) # split by "(dpa)", "(dpa\tmn)" and similar
            city = split[0].strip()
            text = split[1].strip()
        else:
            city = ""
            text = raw_text
        
        if text.startswith("- "): # not very pretty but: remove "- " at the beginning...
            text = text[2:]
        df.loc[ID, 'city'] = city
        
        text = text.replace('\n', ' ').strip() # delete linebreaks
        text = re.sub(' +', ' ', text) # just one space
        df.loc[ID, 'text_preprocessed'] = text
    
    # deleting original 'text' to save space (good?)
    df.drop(columns=['text'], inplace=True)
    
    # adding number of tokens and mean token length (ignoring punctuation)
    # (column "wordcount" already exists, slightly different outcome but okay)
    print("counting tokens...")
    df['nr_tokens'] = [ len([ t for t in nltk.word_tokenize(text) if t not in string.punctuation])
                        for text in df.text_preprocessed ]

    df['mean_token_length'] = [ np.mean([ len(t) for t in nltk.word_tokenize(text) if t not in string.punctuation])
                                for text in df.text_preprocessed ]

    # adding also number of tokens in 'teaser' and 'titelH1'
    df['nr_tokens_teaser'] = [ len([ t for t in nltk.word_tokenize(teaser) if t not in string.punctuation])
                               for teaser in df.teaser ]
    df['nr_tokens_titelH1'] = [ len([ t for t in nltk.word_tokenize(titel) if t not in string.punctuation])
                                for titel in df.titelH1 ]
    
    # adding number of characters of text_preprocessed
    df['nr_char'] = [ len(text) for text in df.text_preprocessed ]

    print("counting sentences...")
    # adding also number of sentences and their mean length (in tokens without punctuation)
    for index, row in df.iterrows():
        text = df.loc[index, 'text_preprocessed']
        sentence_list = nltk.sent_tokenize(text)
        #print(sentence_list)
        df.loc[index, 'nr_sentences'] = len(sentence_list)
        nr_tokens_per_sent = []
        for s in sentence_list:
            token_list = nltk.word_tokenize(s)
            token_list = [t for t in token_list if t not in string.punctuation ] # delete punctuation
            nr_tokens_per_sent.append(len(token_list))
        df.loc[index, 'mean_sentence_length'] = np.mean(nr_tokens_per_sent)
        
    print("averaging avgTimeOnPage per tokens etc...")
    # add a column: avg time divided by wordcount
    df['avgTimeOnPagePerNr_tokens'] = df.avgTimeOnPage/df.nr_tokens

    # add a column: avg time divided by nr_char
    df['avgTimeOnPagePerNr_char'] = df.avgTimeOnPage/df.nr_char
    
    # add a column: pageviews - exits (das sind also die pageviews bei denen time gezählt wird)
    df['pageviews-exits'] = df.pageviews-df.exits
    
    return df


def show_article(ID, df):
    #pageviews = df.loc[ID, 'pageviews']
    #print(f'pageviews: {pageviews}')
    for m in ['pageviews', 'nr_tokens', 'nr_tokens_titelH1', 'nr_tokens_teaser', 'avgTimeOnPagePerNr_tokens', 'stickiness']:
        value = df.loc[ID, m]
        print(f'{m}: {value}')
    for c in ['titelH1', 'teaser', 'text_preprocessed']:
        print('-----')
        print(c)
        print(df.loc[ID, c])
    

def get_set_of_meta_cat(meta_cat):
    tmp_meta = pd.read_csv(OUTPUT / f'meta_file_{meta_cat}.csv', index_col=0)
    columns = tmp_meta.columns.tolist()
    return columns


def get_articles_where(df, meta_cat, label):
    meta = pd.read_csv(OUTPUT / f'meta_file_{meta_cat}.csv', index_col = 0)
    indices = meta.loc[meta[label] == 1].index.tolist()
    return df.loc[indices]

    
    
    
