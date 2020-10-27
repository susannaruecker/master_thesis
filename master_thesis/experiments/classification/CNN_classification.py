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
df = df[['text_preprocessed', 'time_class']] # to save space
print(df.head())

# HYPERPARAMETERS
EPOCHS = 4
BATCH_SIZE = 8
FIXED_LEN = None
MIN_LEN = 500
START = None
LR = 1e-4

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"CNN_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_new"

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
                                                        target = 'time_class', #'time_class',
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
print(data['target'])
print(data['target'].shape)

model = models.CNN(num_outputs = 1,
                   embs_dim = EMBS_DIM,
                   filter_sizes=[3, 4, 5],
                   num_filters=[100,100,100]
                   )

#model = models.CNN_small(num_outputs=1,
#                         embs_dim=EMBS_DIM,
#                         filter_sizes=[3, 4, 5],
#                         num_filters=[30,30,30]) # try a smaller model

model.to(device)

# loss and optimizer
optimizer = optim.AdamW(model.parameters(), lr=LR) # vorher 1e-3 Adam? (lr=1e-5 ist jedenfalls nicht gut!)
loss_fn = nn.BCELoss() # binary cross entropy loss. do we need .to(device) ?
sigmoid_fn = nn.Sigmoid()


##### TRAINING AND EVALUATING #####

### helper function for EVALUATING on dev whenever we want
def evaluate_model(model):
    print("evaluating")
    model = model.eval()
    eval_losses = []

    pred_list = [] # for calculating accuracy
    true_list = []

    with torch.no_grad():
        for nr, d in enumerate(dl_dev):
            print("-Batch", nr, end='\r')
            input_matrix = d["input_matrix"].to(device)
            targets = d["target"].to(device)
            logits = model(input_matrix)
            sigmoids = sigmoid_fn(logits)
            loss = loss_fn(sigmoids, targets)
            eval_losses.append(loss.item())

            targets = targets.squeeze().cpu()
            sigmoids = sigmoids.squeeze().cpu()

            preds = torch.round(sigmoids)  # get "0" or "1" from sigmoids
            true = torch.round(targets)
            pred_list.extend(preds)
            true_list.extend(true)

        nr_correct = np.sum(np.array(pred_list) == np.array(true_list))
        accuracy = nr_correct / len(pred_list)

    return {'eval_loss': np.mean(eval_losses),
            'eval_accuracy' : accuracy}

batch_count = 0
running_loss = []
pred_list = []
true_list = []

for epoch in range(EPOCHS):
    print("Epoch", epoch)

    ### TRAINING on train
    print("training")
    model = model.train()

    for nr, d in enumerate(dl_train):
        print("-Batch", nr, end='\r')
        batch_count += 1

        input_matrix = d["input_matrix"].to(device)
        targets = d["target"].to(device)
        logits = model(input_matrix)
        sigmoids = sigmoid_fn(logits)  # BCE needs sigmoid (ist das richtig?)

        loss = loss_fn(sigmoids, targets)
        running_loss.append(loss.item())
        loss.backward()

        # get the specific 0 or 1 predictions and accuracy
        targets = targets.detach().squeeze().cpu()
        sigmoids = sigmoids.detach().squeeze().cpu()
        preds = torch.round(sigmoids)
        true = torch.round(targets)  # probably not necessary
        pred_list.extend(preds)
        true_list.extend(true)

        if batch_count % 5 == 0: # update only every 5 batches (gradient accumulation) --> simulating bigger "batch size"
            optimizer.step()
            optimizer.zero_grad()

        if batch_count % 200 == 0: # every n batches: write to tensorboard
            # loss
            print(f"running train loss at batch {batch_count}:", np.mean(running_loss))
            writer.add_scalar('train loss', np.mean(running_loss), batch_count)
            running_loss = []

            # accuracy
            nr_correct = np.sum(np.array(pred_list) == np.array(true_list))
            accuracy = nr_correct / len(pred_list)
            print(f"train accuracy at batch {batch_count}:", accuracy)
            writer.add_scalar('train accuracy', accuracy, batch_count)
            pred_list = []
            true_list = []

        if batch_count % 400 == 0: # every n batches: evaluate
            # EVALUATE
            eval_rt = evaluate_model(model = model)
            print("Mean eval loss:", eval_rt['eval_loss'])
            print("Accuracy on dev set:", eval_rt['eval_accuracy'])
            writer.add_scalar('eval loss', eval_rt['eval_loss'], batch_count)
            writer.add_scalar('eval accuracy', eval_rt['eval_accuracy'], batch_count)
            model = model.train() # make sure it is back to train mode


    print("saving model to", model_path)
    torch.save(model.state_dict(), model_path)


print("FIXED_LEN: ", FIXED_LEN)
print("MIN_LEN: ", MIN_LEN)
print("START: ", START)
print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("LR: ", LR)

print(df.shape)