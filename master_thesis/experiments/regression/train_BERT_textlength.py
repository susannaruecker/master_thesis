#!/usr/bin/env python

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import AdamW
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from master_thesis.src import utils, data, models

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
##TODO: Das unterdrückt Warnungen vom Tokenizer, also mit Vorsicht zu genießen

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("device", help="specify which device to use ('cpu' or 'gpu')", type=str)
args = parser.parse_args()

device = torch.device('cpu' if args.device == 'cpu' else 'cuda')
print('Using device:', device)

# get pretrained model and tokenizer from huggingface's transformer library
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'

model = models.BertTextlength(n_outputs=1)


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model.to(device)

# HYPERPARAMETERS
EPOCHS = 40
GPU_BATCH = 4# 32 # what can actually be done in one go on the GPU
BATCH_SIZE = 32 # nr of samples before update step
FIXED_LEN = 512 # 128
MIN_LEN = None # min window size (not used im FIXED_LEN is given)
START = 0 # random, if FIXED_LEN is specified you probably want to start at 0
LR = 1e-5
MASK_WORDS = False
FRACTION = 1

TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ' #'NOZ'

# building identifier from hyperparameters (for Tensorboard and saving model)
starting_time = utils.get_timestamp()
identifier = f"BertTextlength_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_{TARGET}_{PUBLISHER}"

# setting up Tensorboard
if args.device == 'cpu':
    tensorboard_path = utils.TENSORBOARD / f'debugging/{identifier}'
else:
    tensorboard_path = utils.TENSORBOARD / f'runs_{TARGET}/{identifier}_{starting_time}'
    model_path = utils.OUTPUT / 'saved_models' / f'{identifier}_{starting_time}'

writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")


# DataSets and DataLoaders

transform = data.TransformBERT(tokenizer = tokenizer, start = START, fixed_len = FIXED_LEN, min_len= MIN_LEN)
collater = data.CollaterBERT()


ds_train = data.PublisherDataset(publisher=PUBLISHER, set = "train", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_dev = data.PublisherDataset(publisher=PUBLISHER, set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_test = data.PublisherDataset(publisher=PUBLISHER, set = "test", fraction=FRACTION, target  = TARGET, text_base = "article_text", transform = transform)
print("Length of used DataSets:", len(ds_train), len(ds_dev), len(ds_test))

dl_train = DataLoader(ds_train, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater)
dl_dev = DataLoader(ds_dev, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True)

# have a look at one batch in dl_train to see if shapes make sense
d = next(iter(dl_train))
print(d.keys())
input_ids = d['input_ids']
print(input_ids.shape)
attention_mask = d['attention_mask']
print(attention_mask.shape)
print("textlength:", d['textlength'])
print("BERT_tokens:", d['BERT_tokens'])

writer.add_graph(model=model, input_to_model=[input_ids.to(device), attention_mask.to(device), d["BERT_tokens"].to(device)])


# loss and optimizer
optimizer_bert = optim.AdamW(model.bert.parameters(), lr=LR)
optimizer_ffn = optim.AdamW(model.ffn.parameters(), lr=1e-3) #TODO: hier war irgendwie nur Adam, nicht AdamW
#optimizer_bert = AdamW(model.bert.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)
#optimizer_ffn = AdamW(model.ffn.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.01)
loss_fn = nn.MSELoss()  # mean squared error

##### TRAINING AND EVALUATING #####

nr_samples = 0 # counts globally (X-axis for tensorboard etc)
sample_count = 0 # counts up to >=BATCH_SIZE, then update step and back to 0
last_written = 0 # store when last writing/evaluating took place
last_eval = 0
running_loss = []
max_pearson = 0 # initialize max_pearson


for epoch in range(EPOCHS):
    print("Epoch", epoch)

    model.train()
    train_losses = []

    for nr, d in enumerate(dl_train):
        len_minibatch = len(d["target"])
        nr_samples += len_minibatch  # "globally"
        sample_count += len_minibatch  # up to BATCH_SIZE
        print("-Sample", nr_samples, end='\r')

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        textlength = d['BERT_tokens'].to(device)
        #textlength = d['textlength'].to(device)
        targets = d["target"].to(device)
        if MASK_WORDS == True: # masking some of the input ids (better way to do this?
            indices = np.random.choice(np.arange(len(attention_mask.flatten())), replace=False,
                                       size=int(len(attention_mask.flatten()) * 0.3))
            flat = attention_mask.flatten()
            flat[indices] = 0
            attention_mask = flat.reshape(attention_mask.shape)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, textlength=textlength)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if sample_count >= BATCH_SIZE: # after >BATCH_SIZE samples: update step
            optimizer_bert.step()
            optimizer_ffn.step()
            optimizer_bert.zero_grad()
            optimizer_ffn.zero_grad()
            sample_count = 0

        if nr_samples - last_written >= 200:  # every >200 samples: write train loss to tensorboard
            print(f"running train loss at sample {nr_samples}:", np.mean(running_loss))
            writer.add_scalar('train loss', np.mean(running_loss), nr_samples)
            running_loss = []
            last_written = nr_samples

        if nr_samples - last_eval >= 3000:  # every >1000 samples: evaluate
            print("----- start evaluating -----")
            eval_rt = data.evaluate_model(model=model, dl=dl_dev, loss_fn=loss_fn,
                                          using=args.device, max_batch=None)
            last_eval = nr_samples
            # log eval loss and pearson to tensorboard
            print("Mean eval loss:", eval_rt['eval_loss'])
            print("Pearson's r on dev set:", eval_rt['Pearson'])
            print("Spearman's r on dev set:", eval_rt['Spearman'])
            print("MSE on dev set:", eval_rt['MSE'])
            print("MAE on dev set:", eval_rt['MAE'])
            print("RAE on dev set:", eval_rt['RAE'])

            writer.add_scalar('eval loss', eval_rt['eval_loss'], nr_samples)
            writer.add_scalar('Pearson', eval_rt['Pearson'], nr_samples)
            writer.add_scalar('Spearman', eval_rt['Spearman'], nr_samples)
            writer.add_scalar('MSE', eval_rt['MSE'], nr_samples)
            writer.add_scalar('MAE', eval_rt['MAE'], nr_samples)
            writer.add_scalar('RAE', eval_rt['RAE'], nr_samples)

            # save checkpoint if Pearson is bigger than before:
            if eval_rt['Pearson'] >= max_pearson:
                print(f"New best state ({eval_rt['Pearson']}), saving model, optimizer, epoch, sample_count, ... to ",
                      model_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_bert_state_dict': optimizer_bert.state_dict(),
                    'optimizer_ffn_state_dict': optimizer_ffn.state_dict(),
                    'running_loss': running_loss,
                    'nr_samples': nr_samples,
                    'last_eval': last_eval,
                    'last_written': last_written
                    }, model_path)
                max_pearson = eval_rt['Pearson']

            model.train()  # make sure model is back to train mode
            print("----- done evaluating -----")

    print(f"Mean train loss epoch {epoch}:", np.mean(train_losses))
    writer.add_scalar('train loss epoch', np.mean(train_losses), epoch)


print("FIXED_LEN: ", FIXED_LEN)
print("MIN_LEN: ", MIN_LEN)
print("START: ", START)
print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("GPU_BATCH: ", GPU_BATCH)
print("LR: ", LR)
