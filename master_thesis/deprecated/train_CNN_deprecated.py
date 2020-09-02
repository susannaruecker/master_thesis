#!/usr/bin/env python

import torch
from torch import optim, nn
import pandas as pd
import numpy as np
import scipy.stats as st

from master_thesis.src import utils, data, models

assert torch.cuda.is_available()
device = torch.device('cuda:0')
print("Device is: ", device)

# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()

embs = utils.load_fasttext_vectors(limit = None)
EMBS_DIM = 300

# building train-dev-test split, their DataSets and DataLoaders

BATCH_SIZE = 12
MAX_LEN = 500
dl_train, dl_dev, dl_test = data.create_DataLoaders_CNN(df = df,
                                                        target = 'avgTimeOnPagePerNr_tokens',
                                                        text_base = 'text_preprocessed',
                                                        tokenizer = None, # uses default (spacy) tokenizer
                                                        max_len = MAX_LEN,
                                                        batch_size = BATCH_SIZE,
                                                        embs = embs)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_matrix = data['input_matrix']
print(input_matrix.shape)
print(data['target'])
print(data['target'].shape)


#model = models.CNN(num_outputs=1, embs_dim=EMBS_DIM)
model = models.CNN(num_outputs = 1,
                   embs_dim = EMBS_DIM,
                   filter_sizes=[3, 4, 5],
                   num_filters=[100,100,100]
                   )
model.to(device)

# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3) # beim anderen lr=1e-5
                                                     # das Tutorial nutzt optim.Adadelta(cnn_model.parameters(), lr=0.001, rho=0.95)
                                                     # vorher hatte ich AdamW
loss_fn = nn.MSELoss()  # mean squared error

##### TRAINING AND EVALUATING #####

EPOCHS = 2 #15

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

        if nr%50 == 0: # every 50 batches print
            print(f"mean train loss at batch {nr}:", np.mean(train_losses))
    print("Mean train loss epoch:", np.mean(train_losses))

    ### EVALUATING on dev
    print("evaluating")
    model = model.eval()
    eval_losses = []

    pred = []  # for calculating Pearson's r on dev in evaluation per epoch
    true = []

    with torch.no_grad():
        for nr, d in enumerate(dl_dev):
            print("-Batch", nr, end='\r')
            input_matrix = d["input_matrix"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_matrix)
            #print(outputs[:10])
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())

            outputs = outputs.squeeze().cpu()
            targets = targets.squeeze().cpu()
            pred.extend(outputs)
            true.extend(targets)
        print("Mean eval loss:", np.mean(eval_losses))
        print("Pearson's r on dev set:", st.pearsonr(pred, true))

print("saving model")
torch.save(model.state_dict(), utils.OUTPUT / 'saved_models' / f'CNN_{str(MAX_LEN)}.pt')

# to load model again:
#model = models.CNN(num_outputs=1, embs_dim=EMBS_DIM)
#model.load_state_dict(torch.load(utils.OUTPUT / 'saved_models' / f'CNN_{str(MAX_LEN)}.pt)
#model.eval()


print("BATCH_SIZE:", BATCH_SIZE)
print("MAX_LEN: ", MAX_LEN)
print(df.shape)