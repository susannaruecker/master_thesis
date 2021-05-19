#!/usr/bin/env python

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
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
EPOCHS = 30
GPU_BATCH = 1 # what can actually be done in one go on the GPU
BATCH_SIZE = 32 # nr of samples before update step
SECTION_SIZE = 512 #256 #512 #todo: smaller or bigger?
MAX_SECT = 5 # hier waren bei 512 immer 5

LR = 1e-5
MASK_WORDS = False
FRACTION = 1

TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'

# building identifier from hyperparameters (for Tensorboard and saving model)
starting_time = utils.get_timestamp()
identifier = f"BertHierarchicalRNN_SECTIONSIZE{SECTION_SIZE}_MAX_SECT{MAX_SECT}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_{TARGET}_{PUBLISHER}_GRU"


# get pretrained model and tokenizer from huggingface's transformer library
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'

model = models.BertHierarchicalRNN(n_outputs=1, max_sect= MAX_SECT)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model.to(device)


# DataSets and DataLoaders

transform = data.TransformBERT(tokenizer = tokenizer, keep_all = True, start = None, fixed_len = None, min_len= None)
collater = data.CollaterBERT_hierarchical(max_sect=MAX_SECT, section_size=SECTION_SIZE)


ds_train = data.PublisherDataset(publisher=PUBLISHER, set = "train", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_dev = data.PublisherDataset(publisher=PUBLISHER, set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_test = data.PublisherDataset(publisher=PUBLISHER, set = "test", fraction=FRACTION, target  = TARGET, text_base = "article_text", transform = transform)
print("Length of used DataSets:", len(ds_train), len(ds_dev), len(ds_test))

dl_train = DataLoader(ds_train, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater)
dl_dev = DataLoader(ds_dev, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True) # necessary for pred.extend()
dl_test = DataLoader(ds_test, batch_size=GPU_BATCH, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True)

# have a look at one batch in dl_train to see if shapes make sense
d = next(iter(dl_train))
print(d.keys())
input_ids = d['section_input_ids']
print(input_ids.shape)
#print(input_ids)
attention_mask = d['section_attention_mask']
print(attention_mask.shape)
#print(attention_mask)
print("publisher:", d['publisher'])
print("articleId:", d['articleId'])
print("textlength:", d['textlength'])
print("BERT_tokens:", d['BERT_tokens'])


LOAD_CHECKPOINT = True #True # False

# loss and optimizer
optimizer_bert = optim.AdamW(model.bert.parameters(), lr = LR) # bisher LR
optimizer_ffn = optim.AdamW(model.out.parameters(), lr = 1e-3) # bisher 1e-3
optimizer_rnn = optim.AdamW(model.rnn.parameters(), lr = 1e-5) # bisher 1e-5

if LOAD_CHECKPOINT == True:
    optimizer_bert = optim.AdamW(model.bert.parameters(), lr=LR)  # bisher LR
    optimizer_ffn = optim.AdamW(model.out.parameters(), lr=1e-3) # bisher 1e-3
    optimizer_rnn = optim.AdamW(model.rnn.parameters(), lr=1e-5)  # bisher 1e-5

loss_fn = nn.MSELoss()  # mean squared error

if LOAD_CHECKPOINT == True:
    print("using pretrained weights from a BERT baseline")
    identifier = identifier + '_pretrained'
    ### NEW: loading checkpoint (specific layer weights) from a BERT baseline

    checkpoint_path = utils.OUTPUT / 'saved_models' / \
                      'BertFFN_FIXLEN512_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-03-03_00:13:33'
                      #'BertFFN_FIXLEN400_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-05-09_15:54:50'
                      #'BertFFN_FIXLEN256_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-04-22_11:38:31'
                      #'BertFFN_FIXLEN512_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-03-03_00:13:33'
                      #'BertFFN_FIXLEN128_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_splitOptim_2021-02-13_22:43:53'
                      #'BertFFN_FIXLEN128_MINLENNone_START0_EP40_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-02-12_20:24:28'

    model_state_dict = torch.load(checkpoint_path)['model_state_dict'] # nimmt so weniger Speicher in Anspruch ...

    # this compares with the model architecture and deletes/copies over if necessary:
    model_state_dict = utils.modify_state_dict(sd_source=model_state_dict, sd_target=model.state_dict())
    model.load_state_dict(model_state_dict, strict=True)
    print("done with loading checkpoint")

# setting up Tensorboard
if args.device == 'cpu':
    tensorboard_path = utils.TENSORBOARD / f'debugging/{identifier}'
else:
    tensorboard_path = utils.TENSORBOARD / f'runs_{TARGET}/{identifier}_{starting_time}'
    model_path = utils.OUTPUT / 'saved_models' / f'{identifier}_{starting_time}'

writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

##### TRAINING AND EVALUATING #####

nr_samples = 0 # counts globally (X-axis for tensorboard etc)
sample_count = 0 # counts up to >=BATCH_SIZE, then update step and back to 0
last_written = 0 # store when last writing/evaluating took place
last_eval = 0
running_loss = []
max_pearson = 0

for epoch in range(EPOCHS):
    print("Epoch", epoch)

    model = model.train()
    train_losses = []

    for nr, d in enumerate(dl_train):
        len_minibatch = len(d["target"])
        nr_samples += len_minibatch  # "globally"
        sample_count += len_minibatch  # up to BATCH_SIZE
        print("-Sample", nr_samples, end='\r')

        section_input_ids = d["section_input_ids"].to(device)
        section_attention_mask = d["section_attention_mask"].to(device)
        #textlength = d["BERT_tokens"].to(device) #d["textlength"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        outputs = model(section_input_ids = section_input_ids,
                        section_attention_mask = section_attention_mask)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if sample_count >= BATCH_SIZE: # after >BATCH_SIZE samples: update step
            optimizer_bert.step()
            optimizer_ffn.step()
            optimizer_rnn.step()
            optimizer_bert.zero_grad()
            optimizer_ffn.zero_grad()
            optimizer_rnn.zero_grad()
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
                    'optimizer_rnn': optimizer_rnn.state_dict(),
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



print("MAX_SECT: ", MAX_SECT)
print("SECTION_SIZE: ", SECTION_SIZE)
print("EPOCHS: ", EPOCHS)
print("BATCH_SIZE: ", BATCH_SIZE)
print("GPU_BATCH: ", GPU_BATCH)
print("LR: ", LR)

