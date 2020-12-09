#!/usr/bin/env python

import torch
from torch import optim, nn
from transformers import BertTokenizer
import numpy as np
import scipy.stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch.utils.tensorboard import SummaryWriter

from master_thesis.src import utils, data, models

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
##TODO: Das unterdrückt Warnungen vom Tokenizer, also mit Vorsicht zu genießen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# get pretrained model and tokenizer from huggingface's transformer library
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'

model = models.BERT_textlength(n_outputs=1)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

model.to(device)

full = utils.get_raw_df()
df = full
df = df[df.publisher == "NOZ"]
df = df[df.nr_tokens_text >= 100]
df = df[df.nr_tokens_text <= 3000]
df = df[df.avgTimeOnPagePerWordcount <= 3]

df = df.sample(frac=1, replace=False, random_state=1) # take 20% for faster processing # TODO: change back
print(df.head())
print("size of used df:", df.shape)

# HYPERPARAMETERS
EPOCHS = 15
BATCH_SIZE = 8
FIXED_LEN = 400 # random, could be specified (e.g. 400 or 512)
MIN_LEN = None # min window size (not used im FIXED_LEN is given)
START = 0 # random, if MAX_LEN is specified you probably want to start at 0
LR = 1e-5
MASK_WORDS = True

TARGET = 'avgTimeOnPage'

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"BERT_textlength_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_{TARGET}_NOZ_masked"

# setting up Tensorboard
tensorboard_path = f'runs_{TARGET}/{identifier}'
writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

# for saving model after each epoch
model_path = utils.OUTPUT / 'saved_models' / f'{identifier}'

# building train-dev-test split, their DataSets and DataLoaders

window = data.RandomWindow_FFN_BERT(start = START, fixed_len = FIXED_LEN, min_len= MIN_LEN)
collater = data.Collater_FFN_BERT()

dl_train, dl_dev, dl_test = data.create_DataLoaders_FFN_BERT(df=df,
                                                         target = TARGET,
                                                         text_base = 'article_text', #'textPublisher_preprocessed',
                                                         tokenizer = tokenizer,
                                                         train_batch_size = BATCH_SIZE,
                                                         val_batch_size= BATCH_SIZE,
                                                         transform = window,
                                                         collater = collater)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_ids = data['input_ids']
print(input_ids.shape)
attention_mask = data['attention_mask']
print(attention_mask.shape)


# loss and optimizer
# optimizer = optim.AdamW(model.parameters(), lr=LR)
optimizer_bert = optim.AdamW(model.bert.parameters(), lr=LR)
optimizer_ffn = optim.Adam(model.ffn.parameters(), lr=1e-3)
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
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            textlength = d["textlength"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, textlength=textlength)
            # print(outputs[:10])
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())

            outputs = outputs.squeeze().cpu()
            targets = targets.squeeze().cpu()
            pred.extend(outputs)
            true.extend(targets)

    return {'Pearson': st.pearsonr(pred, true)[0],
            'MSE': mean_squared_error(pred, true),
            'MAE': mean_absolute_error(pred, true),
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

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        textlength = d["textlength"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        if MASK_WORDS == True: # masking some of the input ids (better way to do this?
            indices = np.random.choice(np.arange(len(attention_mask.flatten())), replace=False,
                                       size=int(len(attention_mask.flatten()) * 0.3))
            flat = attention_mask.flatten()
            flat[indices] = 0
            attention_mask = flat.reshape(attention_mask.shape)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, textlength=textlength)
        #print(outputs[:10])
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if batch_count % 5 == 0: # update only every n batches (gradient accumulation) --> simulating bigger "batch size"
            #optimizer.step()
            #optimizer.zero_grad()

            optimizer_bert.step()
            optimizer_ffn.step()
            optimizer_bert.zero_grad()
            optimizer_ffn.zero_grad()

        if batch_count % 100 == 0: # every 100 batches: write to tensorboard
            print(f"running train loss at batch {batch_count} (mean over last {len(running_loss)}):", np.mean(running_loss))
            # log the running train loss to tensorboard
            writer.add_scalar('train loss', np.mean(running_loss), batch_count)
            running_loss = []

        if batch_count % 500 == 0: # every 300 batches: evaluate
            # EVALUATE
            eval_rt = evaluate_model(model = model)
            # log eval loss and pearson to tensorboard
            print("Mean eval loss:", eval_rt['eval_loss'])
            print("Pearson's r on dev set:", eval_rt['Pearson'])
            print("MSE on dev set:", eval_rt['MSE'])
            print("MAE on dev set:", eval_rt['MAE'])
            writer.add_scalar('eval loss', eval_rt['eval_loss'], batch_count)
            writer.add_scalar('Pearson', eval_rt['Pearson'], batch_count)
            writer.add_scalar('MSE', eval_rt['MSE'], batch_count)
            writer.add_scalar('MAE', eval_rt['MAE'], batch_count)

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
