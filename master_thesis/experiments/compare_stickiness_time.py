from master_thesis.src import utils
import pandas as pd
import scipy.stats as st


# get raw data
df = pd.read_csv(utils.DATA / 'combined.tsv', sep = '\t')
df = df.fillna('') # replacing Nan with emtpy string
print("Shape of raw df:", df.shape)

# just take articles with ...
df = df.loc[(df['pageviews'] >= 300) &
            (df['nr_tokens'] >= 10) & # to delete articles without text or false text
            (df['avgTimeOnPagePerNr_tokens'] <= 2) &
            (df['avgTimeOnPagePerNr_tokens'] >= 0.1)
            ]
print("Remaining df after conditioning:", df.shape)

print("Pearson: ", st.pearsonr(df.stickiness, df.avgTimeOnPagePerNr_tokens))
# wenn man >=100 pageviews nimmt:
# ca. -0.258
# was sagt das? leicht negative Korrelation?
# also: längere Zeit spricht eher für kleinere stickiness?
# andersrum: kürzere Aufenthaltsdauer spricht eher leicht dafür, dass man auf eine weitere Seite klickt

#TODO: das ist interessant, spricht das jetzt eher für oder gegen die Variable avgTimeOnPage...?

#TODO: mal ein Notebook erstellen, in dem avgTimeOnPagePerNr_tokens betrachtet wird? vielleicht
# auch Standardabweichung etc.und daran abschneiden
