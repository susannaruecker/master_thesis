#!/usr/bin/env python

import torch
from torch import optim, nn
import pandas as pd
import numpy as np

from master_thesis.src import utils, data, models

assert torch.cuda.is_available()
device = torch.device('cuda:0')
print("Device is: ", device)

# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()

embs = utils.load_fasttext_vectors(limit = 1000)

# building train-dev-test split, their DataSets and DataLoaders

BATCH_SIZE = 6
MAX_LEN = 30
dl_train, dl_dev, dl_test = data.create_DataLoaders_CNN(df = df,
                                                        target = 'avgTimeOnPagePerNr_tokens',  # 'avgTimeOnPagePerNr_tokens',
                                                        text_base = 'text_preprocessed',  # 'text_preprocessed', # 'titelH1',
                                                        tokenizer = None, # uses default (spacy) tokenizer
                                                        max_len = MAX_LEN,
                                                        batch_size = BATCH_SIZE,
                                                        embs = embs)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_matrix = data['input_matrix']
print(input_matrix.shape)
#print(data['target'])
print(data['target'].shape)

model = models.CNN(num_outputs=1, fixed_length=MAX_LEN)

# loss and optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
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
        input_matrix = d["input_matrix"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        outputs = model(input_matrix)
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
            input_matrix = d["input_matrix"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_matrix)
            #print(outputs[:10])
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())
        print("Mean eval loss:", np.mean(eval_losses))

#print("saving model")
#model.save_pretrained(utils.OUTPUT / 'saved_models' / f'CNN_{str(MAX_LEN)}')

print("BATCH_SIZE:", BATCH_SIZE)
print("MAX_LEN: ", MAX_LEN)
print(df.shape)