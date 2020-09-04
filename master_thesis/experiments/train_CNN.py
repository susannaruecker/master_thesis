#!/usr/bin/env python

import torch
from torch import optim, nn
import pandas as pd
import numpy as np
import scipy.stats as st
from torch.utils.tensorboard import SummaryWriter

from master_thesis.src import utils, data, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()
df = df[['text_preprocessed', 'avgTimeOnPagePerNr_tokens']] # to save space

# HYPERPARAMETERS
EPOCHS = 4
BATCH_SIZE = 8
FIXED_LEN = None
MIN_LEN = 500 # 400
START = None
LR = 1e-4

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"CNN_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_smaller"

# setting up Tensorboard
tensorboard_path = f'runs/{identifier}'
writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

# for saving model after each epoch
model_path = utils.OUTPUT / 'saved_models' / f'{identifier}'

embs = utils.load_fasttext_vectors(limit = None)
EMBS_DIM = 300

# building train-dev-test split, their DataSets and DataLoaders

window = data.RandomWindow_CNN(start = START, fixed_len = FIXED_LEN, min_len = MIN_LEN)
collater = data.Collater_CNN()

dl_train, dl_dev, dl_test = data.create_DataLoaders_CNN(df = df,
                                                        target = 'avgTimeOnPagePerNr_tokens',
                                                        text_base = 'text_preprocessed',
                                                        tokenizer = None, # uses default (spacy) tokenizer
                                                        embs = embs,
                                                        train_batch_size = BATCH_SIZE,
                                                        val_batch_size= BATCH_SIZE,
                                                        transform = window,
                                                        collater = collater)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_matrix = data['input_matrix']
print(input_matrix.shape)
#print(data['target'])
print(data['target'].shape)

#model = models.CNN(num_outputs = 1,
#                   embs_dim = EMBS_DIM,
#                   filter_sizes=[3, 4, 5],
#                   num_filters=[100,100,100]
#                   )

model = models.CNN_small(num_outputs=1,
                         embs_dim=EMBS_DIM,
                         filter_sizes=[3, 4, 5],
                         num_filters=[50,50,50]) # try a smaller model

model.to(device)

# loss and optimizer
optimizer = optim.AdamW(model.parameters(), lr=LR) # vorher 1e-3 Adam? (lr=1e-5 ist jedenfalls nicht gut!)
#optimizer = optim.Adadelta(model.parameters(), lr=LR, rho=0.95) # den hier hier nutzt das Tutorial vom CNN, war aber bei mir schlecht
loss_fn = nn.MSELoss()  # mean squared error

##### TRAINING AND EVALUATING #####

for epoch in range(EPOCHS):
    print("Epoch", epoch)

    ### TRAINING on train
    print("training")
    model = model.train()
    train_losses = []
    running_loss = 0.0

    for nr, d in enumerate(dl_train):
        print("-Batch", nr, end='\r')
        input_matrix = d["input_matrix"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        outputs = model(input_matrix)
        #print(outputs[:10])
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item()) # for the whole epoch
        running_loss += loss.item() # to get mean over batches
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if nr%50 == 49: # every 50 batches print
            print(f"running train loss batch {nr+1}:", running_loss/50)
            # log the running train loss to tensorboard
            writer.add_scalar('train loss batch',
                              running_loss / 50,
                              epoch * len(dl_train) + nr) # to get the overall batch nr
            running_loss = 0.0

    print("Mean train loss epoch:", np.mean(train_losses))
    writer.add_scalar('train loss epoch', np.mean(train_losses), epoch)

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

        # log eval loss and pearson to tensorboard
        print("Mean eval loss:", np.mean(eval_losses))
        print("Pearson's r on dev set:", st.pearsonr(pred, true))
        writer.add_scalar('eval loss epoch', np.mean(eval_losses), epoch)
        writer.add_scalar('Pearson epoch', st.pearsonr(pred, true)[0], epoch)

    print("saving model to", model_path)
    torch.save(model.state_dict(), model_path)

print("FIXED_LEN: ", FIXED_LEN)
print("MIN_LEN: ", MIN_LEN)
print("START: ", START)
print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("LR: ", LR)

print(df.shape)