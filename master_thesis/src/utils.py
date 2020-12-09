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
from scipy.stats import percentileofscore


#ROOT = Path('/Volumes/INWT/Daten_NLP') # encrypted folder!
ROOT = Path('/home/ruecker/data/Daten_INWT/') # JULIE-Server

#DATA = ROOT / '200820_dataNLP' # alter Datensatz (noch ohne Spalte Ausreißer
#DATA = ROOT / '200921_dataNLP' # neuer
#DATA = ROOT / '201001_dataNLP' # neuer, mit Spalte 'prozentVerlag'
#DATA = ROOT / '201103_dataNLP' # neuer, mit Spalte 'text_crawling'

META = ROOT / 'Dokumentation_Daten.txt'

PROJECT_ROOT = Path(__file__).parents[2].resolve()
OUTPUT = PROJECT_ROOT / 'master_thesis' / 'outputs' # to save tsv files and other stuff

#def read_data(file): # for reading the individual publisher files
#    raw = pd.read_csv(DATA / file, engine='python', quoting=csv.QUOTE_ALL, escapechar = '\\', index_col = 'articleId')
#    #raw = pd.read_csv(DATA / file, encoding='utf-8', error_bad_lines=False, quotechar = '"', doublequote=False)
#    return raw

def get_raw_df():
    #df = pd.read_csv(DATA / 'combined_textCrawling.tsv', sep='\t') # mit dem hier sind die bisherigen gelaufen

    # die hier sind die neuen (mit meinen Texten)
    path_SZ = ROOT / '201112_dataNLP_SZ_TV/201112_SZ_article_text.txt'
    path_TV = ROOT / '201112_dataNLP_SZ_TV/201112_TV_article_text.txt'
    path_NOZ = ROOT / '201117_dataNLP_NOZ/201117_NOZ_article_text.txt'

    SZ = pd.read_csv(path_SZ, sep='\t', index_col = 'articleId')
    TV = pd.read_csv(path_TV, sep='\t', index_col = 'articleId')
    NOZ = pd.read_csv(path_NOZ, sep='\t', index_col='articleId')

    SZ['publisher'] = 'SZ'
    TV['publisher'] = 'TV'
    NOZ['publisher'] = 'NOZ'

    NOZ['titel'] = NOZ["titel_html"]

    NOZ = NOZ[NOZ.other_content == "no"] # die Spalte enthält Hinweis, ob wahrscheinlich Video/Tweet etc. dabei war
    NOZ = NOZ.dropna(subset=['teaser', 'article_body'])  # drop the rows where teaser or article_body is missing

    SZ.rename('SZ_{}'.format, inplace=True)
    TV.rename('TV_{}'.format, inplace=True)
    NOZ.rename('NOZ_{}'.format, inplace=True)

    #print(SZ.head())
    #print(TV.head())
    #print(NOZ.head())

    SZ['nr_tokens_publisher'] = SZ["nr_tokens_text"] # das ist nur temporär, weil so gerade in data.py die Textlänge heißt...
    TV['nr_tokens_publisher'] = TV["nr_tokens_text"]
    NOZ['nr_tokens_publisher'] = NOZ["nr_tokens_text"]


    columns = set(SZ.columns).intersection(TV.columns).intersection(NOZ.columns)
    print("Shared columns:", columns)

    df = pd.concat([SZ[columns], TV[columns], NOZ[columns]])
    # df = df.fillna('')  # replacing Nan with emtpy string
    print("Shape of raw df:", df.shape)
    return df

def get_publisher_df(publ):
    # die hier sind die neuen (mit meinen Texten)
    if publ == "TV":
        path_TV = ROOT / '201112_dataNLP_SZ_TV/201112_TV_article_text.txt'
        TV = pd.read_csv(path_TV, sep='\t', index_col='articleId')
        TV['publisher'] = 'TV'
        TV.rename('TV_{}'.format, inplace=True)
        TV['nr_tokens_publisher'] = TV["nr_tokens_text"]
        df = TV

    if publ == "SZ":
        path_SZ = ROOT / '201112_dataNLP_SZ_TV/201112_SZ_article_text.txt'
        SZ = pd.read_csv(path_SZ, sep='\t', index_col='articleId')
        SZ['publisher'] = 'SZ'
        SZ.rename('SZ_{}'.format, inplace=True)
        SZ['nr_tokens_publisher'] = SZ["nr_tokens_text"]  # nur temporär, weil so gerade in data.py die Textlänge heißt...
        df = SZ

    if publ == "NOZ":
        path_NOZ = ROOT / '201117_dataNLP_NOZ/201117_NOZ_article_text.txt'
        NOZ = pd.read_csv(path_NOZ, sep='\t', index_col='articleId')
        NOZ['publisher'] = 'NOZ'

        NOZ['titel'] = NOZ["titel_html"]

        NOZ = NOZ[NOZ.other_content == "no"]  # die Spalte enthält Hinweis, ob wahrscheinlich Video/Tweet etc. dabei war
        NOZ = NOZ.dropna(subset=['teaser', 'article_body'])  # drop the rows where teaser or article_body is missing
        NOZ.rename('NOZ_{}'.format, inplace=True)
        NOZ['nr_tokens_publisher'] = NOZ["nr_tokens_text"]
        df = NOZ

    print("Shape of df:", df.shape)
    return df


def get_text(publisher, ID):
    if publisher in ['SZ', 'TV']:
        folder = ROOT / '201112_dataNLP_SZ_TV' / publisher / 'txt'
    if publisher == 'NOZ':
        folder = ROOT / '201117_dataNLP_NOZ' / publisher / 'txt'

    with open(folder / f"ID_{ID}.txt", "r") as f:
        text = f.read()
        text = text.replace('\n', ' ').strip()  # delete linebreaks
        text = re.sub(' +', ' ', text)  # just one space
        text = text.replace(u'\xa0', u' ')
        text = text.replace(u'\xad', u'')
    return text


def z_transform(column):
    mean = np.mean(column)
    std = np.std(column)
    z_transformed = [ (value - mean)/std for value in column]
    z_transformed = column.apply(lambda x: (x - mean) / std )

    return z_transformed

def percentile_transform(column):
    percentile_transformed = column.apply(lambda x: percentileofscore(column, x))
    return percentile_transformed


def get_time_class(row, min, max):
    #if min <= row['tokensPerMinute'] <= max:
    if min <= row['avgTimeOnPagePerWordcount'] <= max:
        return 1
    else:
        return 0

def get_conditioned_df(min_diff_pageviews_exits = 50,
                       min_wordcount = 1,
                       max_wordcount = 1000000000, # dummy
                       min_zeilen = 10,
                       max_zeilen = 100,
                       min_avgTimeOnPagePerWordcount = 0.05,
                       max_avgTimeOnPagePerWordcount = 5,
                       min_avgTimeOnPage = 2,  # min 2 seconds
                       max_avgTimeOnPage = 1200, # max 20 minutes
                       min_prozentDpa = 50,
                       min_prozentVerlag = 50,
                       delete_outliers = True,
                       delete_zeilen_mismatch = True, # delete articles where match of zeilen and wordcount ist far off
                       add_time_class = True, # adds a binary time_class label
                       ):


    # get raw data
    df = get_raw_df()

    # conditioning on columns and their values

    df = df.loc[(df['pageviews-exits'] >= min_diff_pageviews_exits) &
                (df['wordcount'] >= min_wordcount) &  # to delete articles without text or erroneous data
                (df['wordcount'] <= max_wordcount) &
                (df['zeilen'] >= min_zeilen) &
                (df['zeilen'] <= max_zeilen) &
                (df['avgTimeOnPage'] >= min_avgTimeOnPage) &
                (df['avgTimeOnPage'] <= max_avgTimeOnPage) &
                (df['avgTimeOnPagePerWordcount'] >= min_avgTimeOnPagePerWordcount) &
                (df['avgTimeOnPagePerWordcount'] <= max_avgTimeOnPagePerWordcount) &
                (df['prozentDpa'] >= min_prozentDpa) &
                (df['prozentVerlag'] >= min_prozentVerlag)
                ]
    if delete_outliers == True:
        df = df.loc[(df['ausreisser'] == 'nein')]

    df = df[df.zeilen > 0]
    df['wordsPerRow'] = df.wordcount / df.zeilen

    if delete_zeilen_mismatch == True:
        avg = np.mean(df.wordsPerRow)
        std = np.std(df.wordsPerRow)
        df = df[(df.wordsPerRow >= avg-(2*std)) & (df.wordsPerRow <= avg+(2*std))]

    if add_time_class == True:
        df['time_class'] = df.apply(lambda row: get_time_class(row=row, min=0.4, max=6), axis=1) # 100, 250

    print("Shape of remaining df after conditioning:", df.shape)
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
      
    # add columns 'city' and 'textDpa_preprocessed'
    print("preprocessing dpa text...")
    for n, ID in enumerate(df.index):
        print(n, end='\r')
        raw_text = df.loc[ID, 'text']
        raw_text = raw_text.replace('\xa0', ' ')
        if "(dpa" in raw_text:
            split = re.split(r'\(dpa\S*', raw_text) # split by "(dpa)", "(dpa\tmn)" and similar
            city = split[0].strip() # take first part as the city
            text = " ".join(split[1:]).strip() # rest is text
                                               # if "Deutsche Presse Agentur (dpa)" appears inside the body of text more splits...
        else:
            city = ""
            text = raw_text
        
        if text.startswith("- "): # not very pretty but: remove "- " at the beginning...
            text = text[2:]
        df.loc[ID, 'city'] = city
        
        text = text.replace('\n', ' ').strip() # delete linebreaks
        text = re.sub(' +', ' ', text) # just one space
        df.loc[ID, 'textDpa_preprocessed'] = text

    # add columns 'textPublisher_preprocessed'
    print("preprocessing publisher text...")
    for n, ID in enumerate(df.index):
        print(n, end='\r')
        raw_text = df.loc[ID, 'textCrawling']
        raw_text = raw_text.replace('\xa0', ' ')
        text = raw_text.replace('\n', ' ').strip()  # delete linebreaks
        text = re.sub(' +', ' ', text)  # just one space
        df.loc[ID, 'textPublisher_preprocessed'] = text

    # deleting original 'text' to save space (good?)
    #df.drop(columns=['text'], inplace=True)
    
    # adding number of tokens and mean token length (ignoring punctuation)
    # (column "wordcount" already exists, slightly different outcome but okay)
    print("counting tokens...")
    df['nr_tokens_dpa'] = [ len([ t for t in nltk.word_tokenize(text) if t not in string.punctuation])
                        for text in df.textDpa_preprocessed ]

    #df['mean_token_length'] = [ np.mean([ len(t) for t in nltk.word_tokenize(text) if t not in string.punctuation])
    #                            for text in df.textDpa_preprocessed ]

    # adding also number of tokens in 'teaser' and 'titelH1'
    #df['nr_tokens_teaser'] = [ len([ t for t in nltk.word_tokenize(teaser) if t not in string.punctuation])
    #                           for teaser in df.teaser ]
    #df['nr_tokens_titelH1'] = [ len([ t for t in nltk.word_tokenize(titel) if t not in string.punctuation])
    #                            for titel in df.titelH1 ]

    df['nr_tokens_publisher'] = [len([t for t in nltk.word_tokenize(text) if t not in string.punctuation])
                       for text in df.textPublisher_preprocessed]

    # adding number of characters of text_preprocessed
    #df['nr_char'] = [ len(text) for text in df.textDpa_preprocessed ]

    #print("counting sentences...")
    # adding also number of sentences and their mean length (in tokens without punctuation)
    #for index, row in df.iterrows():
    #    text = df.loc[index, 'textDpa_preprocessed']
    #    sentence_list = nltk.sent_tokenize(text)
    #    #print(sentence_list)
    #    df.loc[index, 'nr_sentences'] = len(sentence_list)
    #    nr_tokens_per_sent = []
    #    for s in sentence_list:
    #        token_list = nltk.word_tokenize(s)
    #        token_list = [t for t in token_list if t not in string.punctuation ] # delete punctuation
    #        nr_tokens_per_sent.append(len(token_list))
    #    df.loc[index, 'mean_sentence_length'] = np.mean(nr_tokens_per_sent)
        
    print("averaging avgTimeOnPage per tokens etc...")
    # add a column: avg time divided by wordcount
    #df['avgTimeOnPagePerWordcount'] = df.avgTimeOnPage/df.wordcount
    df['avgTimeOnPagePerWordcount'] = df.avgTimeOnPage / df.nr_tokens_publisher

    # add a columns: avg time divided by zeilen
    df['avgTimeOnPagePerRow'] = df.avgTimeOnPage/df.zeilen

    # add a columns: tokens per minute
    #df['tokensPerMinute'] = df.wordcount / df.avgTimeOnPage * 60

    # add a column: avg time divided by nr_char
    #df['avgTimeOnPagePerNr_char'] = df.avgTimeOnPage/df.nr_char
    
    # add a column: pageviews - exits (das sind also die pageviews bei denen time gezählt wird)
    df['pageviews-exits'] = df.pageviews-df.exits
    
    return df


def show_article(ID, df):
    #pageviews = df.loc[ID, 'pageviews']
    #print(f'pageviews: {pageviews}')
    for m in ['pageviews', 'nr_tokens', 'nr_tokens_titelH1', 'nr_tokens_teaser', 'avgTimeOnPagePerWordcount', 'stickiness']:
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

# publisher to int dictionary
publisher_encoding = {'TV': 0,
                      'SZ': 1,
                      'aachener': 2,
                      'bonn': 3,
                      'NOZ': 4}


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
def get_embedding_matrix(text, tokenizer, embs, start = 0, fixed_length=None, min_len = 200):
    if tokenizer is None: # use default (spacy) tokenizer
        nlp = spacy.load("de_core_news_sm", disable=['parser', 'ner'])
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        tokens = tokenizer(text)
        tokens = [ t.text for t in tokens ] # spacy returns objects, not tokens directly
    else: # if another tokenizer is given
        tokens = tokenizer(text)

    original_len = len(tokens)
    if fixed_length:
        window_len = fixed_length
    elif min_len:
        window_len = np.random.randint(low=min_len, high=800)  # random window size between 200 and 800
    else:
        window_len = original_len

        if window_len > original_len:  # just in case text is shorter
            window_len = original_len  # take the original text length

    if start is not None:
        start = start
    elif original_len == window_len:
        start = 0
    else:
        start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
    end = start + window_len

    tokens = tokens[start:end]

    embs_dim = len(embs.get('und')) # take dummy entry to get dimensions of embs
    matrix = np.zeros((window_len, embs_dim))

    for i, t in enumerate(tokens):
        if t in embs:
            vector = embs.get(t)
        else:
            vector = np.zeros(embs_dim)
        matrix[i] = vector

    return matrix # matrix.T ?


# german emotion lexicon (Schmidke et al., 2014), neutral values = [0., 5., 5.], scale valence [-3,3], scale arousal + dominance = [1,9]
def get_ANGST():
    lex = pd.read_excel('/home/ruecker/data/ANGST2014/ratings.xlsx', keep_default_na=False)
    rename ={'VAL_Mean':'valence',
             'ARO_Mean_(ANEW)':'arousal',
             'DOM_Mean': 'dominance',
             'G-word': 'word'}
    lex.rename(columns=rename, inplace=True)
    for c in lex.columns:
        if c not in rename.values():
            lex.drop(columns=c, inplace=True)
    lex.set_index('word', inplace=True)
    lex = lex[~lex.index.duplicated()]
    return lex


# our own lexicon, neutral values = [5., 5., 5.] (zumindest war Warriner so...?)
def get_MEmoLon(max_length=None):
    path = '/home/ruecker/data/MEmoLon/de.tsv'
    lex = pd.read_csv(path, sep='\t', index_col= 0)
    lex = lex[['valence', 'arousal', 'dominance']] # here: just VAD
    if max_length:
        lex = lex[:max_length] # "just" first max_length entries
    return lex


# another german emotion lexicon (VA), (Vo et al., 2009), neutral values = [0., 3.], scale valence = [-3,3], scale arousal, = [1,5]
# Berlin Affective Word List - Reloaded (BAWL-R)
def get_Vo():
    path = '/home/ruecker/data/Vo-2009-Berlin Affective Word List/BAWL-R.csv'
    lex = pd.read_csv(path, sep=';', index_col=1, keep_default_na=False)
    lex.index.rename('word', inplace=True)
    dct = {
        'EMO_MEAN': 'valence',
        'AROUSAL_MEAN': 'arousal'
    }
    lex = lex.rename(columns=dct)
    for c in lex.columns:
        if c not in dct.values():
            lex.drop(columns=c, inplace=True)
    return lex


if __name__ == "__main__":
    test = get_conditioned_df()
    print(test.time_class.value_counts())