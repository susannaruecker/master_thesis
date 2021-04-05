#!/usr/bin/env python

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
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

# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 32
GPU_BATCH = 32
FIXED_LEN = 800 #500
MIN_LEN = None
START = 0
LR = 0.0001
FRACTION = 1

TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'

embs = utils.load_fasttext_vectors(limit = None) # TODO: change to NONE
EMBS_DIM = 300

# building identifier from hyperparameters (for Tensorboard and saving model)
starting_time = utils.get_timestamp()
identifier = f"CNN_FIXLEN{FIXED_LEN}_MINLEN{MIN_LEN}_START{START}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_{TARGET}_{PUBLISHER}_num_filters_64"

# setting up Tensorboard
if args.device == 'cpu':
    tensorboard_path = utils.TENSORBOARD / f'debugging/{identifier}'
else:
    tensorboard_path = utils.TENSORBOARD / f'runs_{TARGET}/{identifier}_{starting_time}'
    model_path = utils.OUTPUT / 'saved_models' / f'{identifier}_{starting_time}'

writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")


model = models.CNN(n_outputs=1, embs_dim=EMBS_DIM)
model.to(device)

# DataSets and DataLoaders

transform = data.TransformCNN(tokenizer = None, embs = embs, start = START, fixed_len = FIXED_LEN, min_len = MIN_LEN)
collater = data.CollaterCNN()

ds_train = data.PublisherDataset(publisher="NOZ", set = "train", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_dev = data.PublisherDataset(publisher="NOZ", set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_test = data.PublisherDataset(publisher="NOZ", set = "test", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
print("Length of used DataSets:", len(ds_train), len(ds_dev), len(ds_test))

dl_train = DataLoader(ds_train, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater)
dl_dev = DataLoader(ds_dev, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True)

# have a look at one batch in dl_train to see if shapes make sense
d = next(iter(dl_train))
print(d.keys())
input_matrix = d['input_matrix']
print(input_matrix.shape)
print(d['target'].shape)

# loss and optimizer
#optimizer = optim.Adam(model.parameters(), lr=LR) # vorher 1e-3 Adam? (lr=1e-5 ist jedenfalls nicht gut!)
optimizer = optim.AdamW(model.parameters(), lr=LR) # vorher 1e-3 Adam? (lr=1e-5 ist jedenfalls nicht gut!)

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

        input_matrix = d["input_matrix"].to(device)
        targets = d["target"].to(device)
        outputs = model(input_matrix=input_matrix)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if sample_count >= BATCH_SIZE: # after >BATCH_SIZE samples: update step
            optimizer.step()
            optimizer.zero_grad()
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
                    'optimizer_state_dict': optimizer.state_dict(),
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
