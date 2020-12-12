#!/usr/bin/env python

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
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

model = models.baseline_textlength(n_outputs=1)
model.to(device)

# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 10
LR = 1e-3 # 1e-3
FRACTION = 1

TARGET = 'avgTimeOnPage'

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"baseline_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_{TARGET}_NOZ"

# setting up Tensorboard
tensorboard_path = f'runs_{TARGET}/{identifier}'
writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

# for saving model after each epoch
model_path = utils.OUTPUT / 'saved_models' / f'{identifier}'

# DataSets and DataLoaders

ds_train = data.PublisherDataset(publisher="NOZ", set = "train", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = None)
ds_dev = data.PublisherDataset(publisher="NOZ", set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = None)
ds_test = data.PublisherDataset(publisher="NOZ", set = "test", fraction=FRACTION, target  = TARGET, text_base = "article_text", transform = None)
print("Length of used DataSets:", len(ds_train), len(ds_dev), len(ds_test))

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=None)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=None)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=None)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())


# loss and optimizer
#optimizer_ffn_embs = optim.Adam(list(model.ffn.parameters()) + list(model.publisher_embs.parameters()), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR)
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
            textlength = d["textlength"].to(device)
            #publisher = d["publisher"].to(device)
            targets = d["target"].to(device)
            outputs = model(textlength=textlength)#, publisher=publisher)
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

        textlength = d["textlength"].to(device)
        #publisher = d["publisher"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        outputs = model(textlength=textlength)#, publisher=publisher)
        #print(outputs[:10])
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if batch_count % 5 == 0: # update only every n batches (gradient accumulation) --> simulating bigger "batch size"
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


print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("LR: ", LR)


