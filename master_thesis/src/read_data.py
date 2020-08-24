#!/usr/bin/env python

import pandas as pd
from master_thesis.src import utils

# Reading in data files
bonn = utils.read_data('200820_Bonn.txt')                  # Generalanzeiger Bonn
NOZ = utils.read_data('200820_NOZ.txt')                    # Neue Osnabrücker Zeitung
aachener = utils.read_data('200820_aachener_zeitung.txt')  # Aachener Zeitung
SZ = utils.read_data('200820_SZ.txt')                      # Saarbrücker Zeitung
TV = utils.read_data('200820_TV.txt')                      # Trierscher Volksfreund

#aachener0 = utils.read_data('200707_aachener_zeitung.txt')
# alle Artikel aus aachener0 sind auch in aachener, die Werte dort sind aber aktueller

print(bonn.shape, NOZ.shape, aachener.shape, SZ.shape, TV.shape)

# Before concatenation: add a column with info about which publisher
bonn['publisher'] = 'bonn'
NOZ['publisher'] = 'NOZ'
aachener['publisher'] = 'aachener'
SZ['publisher'] = 'SZ'
TV['publisher'] = 'TV'

# asserting that all have same columns/shape
#bonn.columns == NOZ.columns
#NOZ.columns == aachener.columns
#aachener.columns == SZ.columns
#SZ.columns == TV.columns

# concatenating all
combined = pd.concat([bonn, NOZ, aachener, SZ, TV], axis=0)
print(combined.shape)

#i_list = combined.index
#i_set = set(i_list)
#print(len(i_list))
#print(len(i_set)) # all different IDs, good!

#combined.info()
#combined = combined.fillna('') # replacing Nan with emtpy string

# bei 2 fehlt 'text'
print(combined[combined['text'].isnull()])
#combined.loc[1986628].teaser # vielleicht der Artikeltext fälschlicherweise als Teaser?
#combined.loc[2015422]        # Text fehlt tatsächlich

# creating dict with descriptions of columns from meta file
meta_dict = utils.create_meta_dict()

# deleting colums with 'tbi' in meta_dict
use_columns = [var for var in meta_dict.keys() if meta_dict[var] != 'tbi' and var != 'articleId']
combined = combined[use_columns]

print(combined.shape)

combined['text'].fillna('', inplace = True) # damit die zwei leeren Texte keine Probleme bereiten

# adding columns with minimal preprocessed text, nr_tokens, nr_sentences, ...

combined = utils.add_meta_columns(combined)

# saving to disk!
combined.to_csv(utils.OUTPUT / 'combined.tsv', sep = '\t', index = True)
print(combined.shape)
