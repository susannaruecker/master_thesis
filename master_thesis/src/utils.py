import pandas as pd
import numpy as np
import string
import nltk
#nltk.download('punkt')
import json
import re
import csv
from pathlib import Path
import spacy
import wget
import gzip
import os
import os.path

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

def get_conditioned_df(min_pageviews = 100,
                       max_pageviews = 1000000000, # dummy max_pageviews
                       min_nr_tokens = 10,
                       max_nr_tokens = 1000000000, # dummy max_tokens
                       min_time = 0.1,
                       max_time =  2):

    # get raw data
    df = pd.read_csv(DATA / 'combined.tsv', sep='\t')
    #df = df.fillna('')  # replacing Nan with emtpy string
    print("Shape of raw df:", df.shape)

    # conditioning on columns and their values

    df = df.loc[(df['pageviews'] >= min_pageviews) &
                (df['pageviews'] <= max_pageviews) &
                (df['nr_tokens'] >= min_nr_tokens) &  # to delete articles without text or erroneous data
                (df['nr_tokens'] <= max_nr_tokens) &  # to delete articles without text or erroneous data
                (df['avgTimeOnPagePerNr_tokens'] <= max_time) &  # hier war vorher 4
                (df['avgTimeOnPagePerNr_tokens'] >= min_time)  # hier war vorher 0.01
                ]
    print("Remaining df after conditioning:", df.shape)
    return df


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
    #df.drop(columns=['text'], inplace=True)
    
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


#class Preprocessor():
#    def __init__(self):
#        self.nlp = spacy.load("de_core_news_sm", disable=['parser', 'ner'])
#        #self.nlp = spacy.load("de_core_news_md", disable=['parser', 'ner'])
#
#    def __call__(self, doc):
#        rt = []
#        doc = self.nlp(doc)
#        for token in doc:
#            rt.append(token.lemma_.lower()) # stopwords stay in, (skleans CountVectorizer deletes it later)
#
#        return " ".join(rt)


class Preprocessor():
    def __init__(self, delete_stopwords=False, lemmatize=False, delete_punctuation=False):
        self.nlp = spacy.load("de_core_news_sm", disable=['parser', 'ner'])
        # self.nlp = spacy.load("de_core_news_md", disable=['parser', 'ner'])
        self.delete_stopwords = delete_stopwords
        self.lemmatize = lemmatize
        self.delete_punctuation = delete_punctuation
        self.stopwords = nltk.corpus.stopwords.words('german')

    def __call__(self, doc):
        rt = []
        doc = self.nlp(doc)

        if self.lemmatize == True:
            for token in doc:
                rt.append(token.lemma_.lower())
        else:
            for token in doc:
                rt.append(token.text)

        if self.delete_punctuation == True:
            rt = [t for t in rt if t not in string.punctuation]

        if self.delete_stopwords == True:
            if self.lemmatize == True:
                self.stopwords = [self.nlp(s)[0].lemma_ for s in self.stopwords]
            rt = [t for t in rt if t not in self.stopwords]

        return " ".join(rt)


def load_fasttext_vectors(limit=None):
    # if necessary download german fastText Embeddings
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz'

    # target_path = '/Users/Sanna/data/cc.de.300.vec.gz'                 # Laptop
    target_path = '/home/ruecker/data/fasttext_vectors/cc.de.300.vec.gz' # JULIE

    if not os.path.isfile(target_path):
        print('downloading vectors...')
        wget.download(url, out=target_path)
        print('done')
    else:
        print('file already exists')

    print('loading embeddings ...')
    f = gzip.open(target_path, 'rb')
    n, d = map(int, f.readline().split())
    data = {}
    counter = 0
    for line in f:
        line = line.decode()
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
        if limit:
            if counter >= limit:
                break
            else:
                counter += 1
    f.close()
    if limit:
        print(f"done loading embs WITH LIMIT {str(limit)}")
    else:
        print('done loading embs FULL')
    return data


def get_averaged_vector(text, preprocessor, embs):
    vector = np.zeros(300)
    text_preprocessed = preprocessor(text)
    tokens = text_preprocessed.split() # preprocessor returns string with " " as separator, so needs to be split up
    counter = 0
    for t in tokens:
        #print(t)
        if t in embs:
            vector += embs.get(t)
            counter +=1
    #print(counter)
    if counter !=0:
        vector = vector/counter
    return vector


# feature extraction for CNN: matrix with embedding of tokens, padded/trimmed to fixed_len if given
def get_embedding_matrix(text, tokenizer, embs, fixed_length=None):
    if tokenizer is None: # use default (spacy) tokenizer
        nlp = spacy.load("de_core_news_sm", disable=['parser', 'ner'])
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        tokens = tokenizer(text)
        tokens = [ t.text for t in tokens ] # spacy returns objects, not tokens directly
    else: # if another tokenizer is given
        tokens = tokenizer(text)

    length = len(tokens)

    embs_dim = len(embs.get('und')) # take dummy entry to get dimensions of embs
    if fixed_length:
        matrix = np.zeros((fixed_length, embs_dim))
    else:
        matrix = np.zeros((length, embs_dim))

    for i, t in enumerate(tokens):
        if fixed_length and i >= fixed_length:
            break
        if t in embs:
            vector = embs.get(t)
        else:
            vector = np.zeros(embs_dim)
        matrix[i] = vector

    return matrix # matrix.T ?

