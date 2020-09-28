#!/usr/bin/env python

import torch
from torch import optim, nn
from transformers import BertTokenizer, DistilBertTokenizer
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

MODEL = 'BERTModel' #'BERTAvg' #'BERTAvg' #'BERTModel'

if MODEL == 'BERT':
    model = models.Bert_sequence(n_outputs=1)       # this is exactly BertForSequenceClassifiaction (but just outputs logits)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
elif MODEL == 'BERTModel':
    model = models.Bert_regression(n_outputs=1)     # this is customized (dropout!) BertModel
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
elif MODEL == 'BERTAvg':                            # this uses averaged last hidden states over sequence instead of CLS-token
    model = models.Bert_averaging(n_outputs=1)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#if MODEL == 'DistilBERT':
#    model = models.DistilBert_sequence(n_outputs=1)     # this ist DistilBert's Sequence Classification
#    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')

model.to(device)

# get data (already conditionend on min_pageviews etc)
df = utils.get_conditioned_df()
df = df[['text_preprocessed', 'time_class']]
print(df.head())

# HYPERPARAMETERS
EPOCHS = 3
BATCH_SIZE = 8
FIXED_LEN = None # random, could be specified (e.g. 400 or 512)
MIN_LEN = 500 # min window size (not used im FIXED_LEN is given)
START = None # random, if MAX_LEN is specified you probably want to start at 0
LR = 1e-6 # before it was 1e-5

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"{MODEL}_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_gradient_acc_new"

# setting up Tensorboard
tensorboard_path = f'runs/{identifier}'
writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

# for saving model after each epoch
model_path = utils.OUTPUT / 'saved_models' / 'classification' / f'{identifier}'

# building train-dev-test split, their DataSets and DataLoaders

window = data.RandomWindow_BERT(start = START, fixed_len = FIXED_LEN, min_len= MIN_LEN)
collater = data.Collater_BERT()

dl_train, dl_dev, dl_test = data.create_DataLoaders_BERT(df=df,
                                                         target = 'time_class', #  'time_class',
                                                         text_base = 'text_preprocessed',
                                                         tokenizer = tokenizer,
                                                         train_batch_size = BATCH_SIZE,
                                                         val_batch_size= BATCH_SIZE,
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
print(data['target'])


# loss and optimizer
optimizer = optim.AdamW(model.parameters(), lr=LR)

loss_fn = nn.BCELoss() # binary cross entropy loss. do we need .to(device) ?
sigmoid_fn = nn.Sigmoid() # TODO: das ist schon richtig, dass man das vor BCE selbst machen muss, oder?

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
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
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

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
