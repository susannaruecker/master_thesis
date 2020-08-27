import torch
from transformers import BertTokenizer, BertForSequenceClassification
from master_thesis.src import utils, models, data
import scipy.stats as st
import pandas as pd

assert torch.cuda.is_available()
device = torch.device('cuda:0')
print("Device is: ", device)

# load the trained and saved model
# tokenizer is the unchanged one from huggingface
MAX_LEN = 300

model, tokenizer = models.get_model_and_tokenizer(utils.OUTPUT / 'saved_models' / f'BERT_{str(MAX_LEN)}')
model.to(device)

# get raw data
df = pd.read_csv(utils.DATA / 'combined.tsv', sep = '\t')
df = df.fillna('') # replacing Nan with emtpy string
print("Shape of raw df:", df.shape)

# just take articles with ...
df = df.loc[(df['pageviews'] >= 100) & # hier war vorher 20
            #(df['publisher'] == 'bonn') & # das hier war für weniger Daten zum Fehlerfinden
            (df['nr_tokens'] >= 10) &  # to delete articles without text or false text
            (df['avgTimeOnPagePerNr_tokens'] <= 2) & # hier war vorher 4
            (df['avgTimeOnPagePerNr_tokens'] >= 0.1) # hier war vorher 0.01
            ]
print("Remaining df after conditioning:", df.shape)


# load Dataloader for dev-Set (batch size in interference can be big, no calculation needed)
BATCH_SIZE = 200
_, dl_dev, _ = data.create_DataLoaders(df = df,
                                       target = 'avgTimeOnPagePerNr_tokens', #'avgTimeOnPagePerNr_tokens',
                                       text_base = 'text_preprocessed',# text_preprocessed',
                                       tokenizer = tokenizer,
                                       max_len = MAX_LEN,
                                       batch_size = BATCH_SIZE)

pred = []
true = []

model.eval()
with torch.no_grad():
    for d in dl_dev:
        input_ids = d["input_ids"].to(device)
        #print(input_ids)
        attention_mask = d["attention_mask"].to(device)
        #print(attention_mask)
        pred_dev = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        #print(pred_dev[:10])
        y_dev = d["target"].to(device)

        pred_dev = pred_dev.squeeze().cpu()
        y_dev = y_dev.squeeze().cpu()

        print("Pearson's of this batch:", st.pearsonr(pred_dev, y_dev))

        pred.extend(pred_dev)
        true.extend(y_dev)

print(len(pred))
print(len(true))
print("Pearson's r on whole dev set:", st.pearsonr(pred, true))
# bei max_len 100 (0.46954919345639684, 7.58807236498442e-71) also ganz okay, ähnlich wie BOW
# bei max_len 200 (0.5467132606724006, 3.380111928800224e-100) also ziemlich viel besser
# bei max_len 300 (0.5927131137308533, 1.0772004471333138e-121)s

#TODO: true und pred plotten