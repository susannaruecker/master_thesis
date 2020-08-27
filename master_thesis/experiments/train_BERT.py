#!/usr/bin/env python

import torch
from torch import optim, nn
import pandas as pd
import numpy as np

from master_thesis.src import utils, data, models

assert torch.cuda.is_available()
device = torch.device('cuda:0')
print("Device is: ", device)

# get pretrained model and tokenizer from huggingface's transformer library
model, tokenizer = models.get_model_and_tokenizer()
model.to(device)

# try out tokenizer
#sample_text = "Das hier ist ein deutscher Beispieltext. Und einen zweiten müssen wir auch noch haben."
#tokens = tokenizer.tokenize(sample_text) # just tokenizes
#token_ids = tokenizer.convert_tokens_to_ids(tokens)
#ids = tokenizer.encode(sample_text) # already adds special tokens
#encoded_plus = tokenizer.encode_plus(sample_text,
#                                     max_length = 10,
#                                     return_token_type_ids=False,
#                                     pad_to_max_length=True,
#                                     truncation=True,
#                                     return_attention_mask=True,)

#print(tokens)
#print(token_ids)
#print(ids)
#print("Testing the tokenizer:" , encoded_plus)

#tokenizer.get_vocab() # shows tokenizer vocab (subwords!)
#tokenizer.sep_token, tokenizer.sep_token_id, tokenizer.cls_token, tokenizer.cls_token_id, tokenizer.pad_token, tokenizer.pad_token_id


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

# building train-dev-test split, their DataSets and DataLoaders

BATCH_SIZE = 6
MAX_LEN = 512
dl_train, dl_dev, dl_test = data.create_DataLoaders_BERT(df = df,
                                                         target = 'avgTimeOnPagePerNr_tokens',  # 'avgTimeOnPagePerNr_tokens',
                                                         text_base = 'text_preprocessed',  # 'text_preprocessed', # 'titelH1',
                                                         tokenizer = tokenizer,
                                                         max_len = MAX_LEN,
                                                         batch_size = BATCH_SIZE)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_ids = data['input_ids']
#print(input_ids)
print(input_ids.shape)
attention_mask = data['attention_mask']
#print(attention_mask)
print(attention_mask.shape)
print(data['target'].shape)

# loss and optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

#LEARNING_RATE = 0.001 #0.001 # 0.00001
#optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()  # mean squared error

##### TRAINING AND EVALUATING #####

EPOCHS = 5 #15

for epoch in range(EPOCHS):
    print("Epoch", epoch)

    ### TRAINING on train
    print("training")
    model = model.train()
    train_losses = []

    for nr, d in enumerate(dl_train):
        print("-Batch", nr, end='\r')
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]  # stimmt das so? ist [0] die logits?
        #print(outputs[:10])
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if nr%50 == 0: # every 100 batches print
            print(f"mean train loss at batch {nr}:", np.mean(train_losses))
    print("Mean train loss epoch:", np.mean(train_losses))

    ### EVALUATING on dev
    print("evaluating")
    model = model.eval()
    eval_losses = []

    with torch.no_grad():
        for nr, d in enumerate(dl_dev):
            print("-Batch", nr, end='\r')
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]  # stimmt das so?
            #print(outputs[:10])
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())
        print("Mean eval loss:", np.mean(eval_losses))

#print("saving model")
model.save_pretrained(utils.OUTPUT / 'saved_models' / f'BERT_{str(MAX_LEN)}')

print("BATCH_SIZE:", BATCH_SIZE)
print("MAX_LEN: ", MAX_LEN)
print(df.shape)