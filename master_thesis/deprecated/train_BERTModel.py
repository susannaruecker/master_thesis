#!/usr/bin/env python

### Als Gedanken-Stütze
### Der Unterschied zu train_BERT ist: Hier verwende ich das originale Bert (siehe models)
### mit eigenem Dropout und zwei linearen Layern
### statt dem vorgefertigten BertForSequenceClassification
### Grund: ich habe das Gefühl, dass das andere vielleicht zu sehr überfittet

import torch
from torch import optim, nn
from transformers import BertTokenizer
import numpy as np
import scipy.stats as st
from torch.utils.tensorboard import SummaryWriter

from master_thesis.src import utils, data, models

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
#TODO: Das unterdrückt Warnungen vom Tokenizer, also mit Vorsicht zu genießen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# get pretrained model and tokenizer from huggingface's transformer library
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = models.Bert_regression(n_outputs = 1)
model.to(device)

# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()
df = df[['text_preprocessed', 'avgTimeOnPagePerNr_tokens']] # to save space

# HYPERPARAMETERS
EPOCHS = 5
BATCH_SIZE = 8
FIXED_LEN = None # random, could be specified (e.g. 400 or 512)
MIN_LEN = 500 # min window size (not used im FIXED_LEN is given)
START = None # random, if MAX_LEN is specified you probably want to start at 0
LR = 1e-4 # before it was 1e-5

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"BERTModel_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}"

# setting up Tensorboard
tensorboard_path = f'runs/{identifier}'
writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

# for saving model after each epoch
model_path = utils.OUTPUT / 'saved_models' / f'{identifier}'

# building train-dev-test split, their DataSets and DataLoaders

window = data.RandomWindow_BERT(start = START, fixed_len = FIXED_LEN, min_len= MIN_LEN)
collater = data.Collater_BERT()

dl_train, dl_dev, dl_test = data.create_DataLoaders_BERT(df=df,
                                                         target = 'avgTimeOnPagePerNr_tokens',
                                                         text_base = 'text_preprocessed',
                                                         tokenizer = tokenizer,
                                                         train_batch_size = BATCH_SIZE,
                                                         val_batch_size = BATCH_SIZE,
                                                         transform = window,
                                                         collater = collater)

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
optimizer = optim.AdamW(model.parameters(), lr=LR)
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
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)
        #print("Target shape", targets.shape)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #print(outputs[:10])
        #print("Output shape:", outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss += loss.item()
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if nr % 50 == 49:  # every 50 batches print
            print(f"running train loss batch {nr + 1}:", running_loss / 50)
            # log the running train loss to tensorboard
            writer.add_scalar('train loss batch',
                              running_loss / 50,
                              epoch * len(dl_train) + nr)  # to get the overall batch nr

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
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)
            #print("Eval target shape", targets.shape)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            #print(outputs[:10])
            #print("Eval outputs shape", outputs.shape)
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
    #model.save_pretrained(model_path)
    torch.save(model.state_dict(), model_path)

print("FIXED_LEN: ", FIXED_LEN)
print("MIN_LEN: ", MIN_LEN)
print("START: ", START)
print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("LR: ", LR)

print(df.shape)
