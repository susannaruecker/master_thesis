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
df = df[['text_preprocessed', 'avgTimeOnPagePerWordcount']] # to save space

# HYPERPARAMETERS
EPOCHS = 5
BATCH_SIZE = 8
FIXED_LEN = 500 # None
MIN_LEN = 500#500 # 400
START = 0 #None
LR = 1e-4

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"CNN_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_gradient_acc_smaller_new"

# setting up Tensorboard
tensorboard_path = f'runs2/{identifier}'
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
                                                        target = 'avgTimeOnPagePerWordcount',
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

### helper function for EVALUATING on dev whenever we want
def evaluate_model(model):
    print("evaluating")
    model = model.eval()
    eval_losses = []

    pred = []  # for calculating Pearson's r on dev
    true = []

    with torch.no_grad():
        for nr, d in enumerate(dl_dev):
            print("-Batch", nr, end='\r')
            input_matrix = d["input_matrix"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_matrix)
            # print(outputs[:10])
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())

            outputs = outputs.squeeze().cpu()
            targets = targets.squeeze().cpu()
            pred.extend(outputs)
            true.extend(targets)

    return {'Pearson': st.pearsonr(pred, true)[0],
            'eval_loss': np.mean(eval_losses)}


batch_count = 0
running_loss = []

for epoch in range(EPOCHS):
    print("Epoch", epoch)

    ### TRAINING on train
    print("training")
    model = model.train()
    train_losses = []

    for nr, d in enumerate(dl_train):
        print("-Batch", nr, end='\r')
        batch_count += 1

        input_matrix = d["input_matrix"].to(device)
        targets = d["target"].to(device)
        outputs = model(input_matrix)
        #print(outputs[:10])
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if batch_count % 2 == 0: # vorher % 5 # update only every 5 batches (gradient accumulation) --> simulating bigger "batch size"
            optimizer.step()
            optimizer.zero_grad()

        if batch_count % 100 == 0: # every 100 batches: write to tensorboard
            print(f"running train loss at batch {batch_count} (mean over last {len(running_loss)}):", np.mean(running_loss))
            # log the running train loss to tensorboard
            writer.add_scalar('train loss', np.mean(running_loss), batch_count)
            running_loss = []

        if batch_count % 300 == 0: # every 300 batches: evaluate
            # EVALUATE
            eval_rt = evaluate_model(model = model)
            # log eval loss and pearson to tensorboard
            print("Mean eval loss:", eval_rt['eval_loss'])
            print("Pearson's r on dev set:", eval_rt['Pearson'])
            writer.add_scalar('eval loss', eval_rt['eval_loss'], batch_count)
            writer.add_scalar('Pearson', eval_rt['Pearson'], batch_count)
            model = model.train() # make sure it is back to train mode

    print("Mean train loss epoch:", np.mean(train_losses))
    writer.add_scalar('train loss epoch', np.mean(train_losses), epoch)

    print("saving model to", model_path)
    torch.save(model.state_dict(), model_path)


print("FIXED_LEN: ", FIXED_LEN)
print("MIN_LEN: ", MIN_LEN)
print("START: ", START)
print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("LR: ", LR)

print(df.shape)
