#!/usr/bin/env python

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
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


# HYPERPARAMETERS
EPOCHS = 65
BATCH_SIZE = 1
SECTION_SIZE = 512 #todo: change back to 150 or higher?
MAX_SECT = 4
LR = 1e-5
MASK_WORDS = False
FRACTION = 1

TARGET = 'avgTimeOnPage'
PUBLISHER = 'NOZ'

# building identifier from hyperparameters (for Tensorboard and saving model)
identifier = f"BERT_hierarchical_SECTIONSIZE{SECTION_SIZE}_MAX_SECT{MAX_SECT}_EP{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_{TARGET}_{PUBLISHER}_pretrained_BERT_baseline"

# setting up Tensorboard
tensorboard_path = f'runs_{TARGET}/{identifier}'
writer = SummaryWriter(tensorboard_path)
print(f"logging with Tensorboard to path {tensorboard_path}")

# get pretrained model and tokenizer from huggingface's transformer library
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'

model = models.BERT_hierarchical(n_outputs=1, max_sect= MAX_SECT)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model.to(device)

# for saving model after each epoch
model_path = utils.OUTPUT / 'saved_models' / f'{identifier}'

# DataSets and DataLoaders

transform = data.TransformBERT(tokenizer = tokenizer, keep_all = True, start = None, fixed_len = None, min_len= None)
collater = data.CollaterBERT_hierarchical(max_sect=MAX_SECT, section_size=SECTION_SIZE)


ds_train = data.PublisherDataset(publisher=PUBLISHER, set = "train", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_dev = data.PublisherDataset(publisher=PUBLISHER, set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_test = data.PublisherDataset(publisher=PUBLISHER, set = "test", fraction=FRACTION, target  = TARGET, text_base = "article_text", transform = transform)
print("Length of used DataSets:", len(ds_train), len(ds_dev), len(ds_test))

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=collater)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True) # necessary for pred.extend()
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=collater, drop_last=True)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_ids = data['section_input_ids']
print(input_ids.shape)
#print(input_ids)
attention_mask = data['section_attention_mask']
print(attention_mask.shape)
#print(attention_mask)
print(data['publisher'])
print(data['textlength'])
print(data['articleId'])



# loss and optimizer
#optimizer = optim.AdamW(model.parameters(), lr=LR)
optimizer_bert = optim.AdamW(list(model.bert.parameters())
                             +list(model.bert_ffn.parameters()))
optimizer_weight_vector = optim.AdamW([model.weight_vector], lr=1e-3)
optimizer_ffn = optim.AdamW(model.ffn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # mean squared error


### NEW: loading checkpoint (specific layer weights) from bert baseline

checkpoint_path = utils.OUTPUT / 'saved_models' / 'BERT_baseline_FIXLEN512_MINLENNone_START0_EP20_BS5_LR1e-05_avgTimeOnPage_NOZ'
#checkpoint_path = utils.OUTPUT / 'saved_models' / 'BERT_textlength_baseline_FIXLEN512_MINLENNone_START0_EP30_BS5_LR1e-05_avgTimeOnPage_NOZ'

model_state_dict = torch.load(checkpoint_path)['model_state_dict'] # nimmt so weniger Speicher in Anspruch ...

# this compares with the model architecture and deletes/copies over if necessary:
model_state_dict = utils.modify_state_dict(sd_source=model_state_dict, sd_target=model.state_dict())
model.load_state_dict(model_state_dict, strict=True)
print("done with loading checkpoint")



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
            section_input_ids = d["section_input_ids"].to(device)
            section_attention_mask = d["section_attention_mask"].to(device)
            textlength = d["textlength"].to(device)
            targets = d["target"].to(device)
            outputs = model(section_input_ids=section_input_ids,
                            section_attention_mask=section_attention_mask,
                            textlength=textlength)
            # print(outputs[:10])
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())

            outputs = outputs.squeeze().cpu()
            targets = targets.squeeze().cpu()

            if BATCH_SIZE==1: # necessary if BATCH_SIZE = 1
                pred.append(outputs)
                true.append(targets)
            else:
                pred.extend(outputs)
                true.extend(targets)

    rand_int = np.random.randint(low = 0, high = len(pred)-20)

    print("Inspecting some predicted and their true values:")
    print("predicted:", [ round(t.item(), 2) for t in pred[rand_int:rand_int+10] ])
    print("true:", [ round(t.item(), 2) for t in true[rand_int:rand_int+10] ])

    return {'Pearson': st.pearsonr(pred, true)[0],
            'MSE': mean_squared_error(pred, true),
            'MAE': mean_absolute_error(pred, true),
            'RAE': utils.relative_absolute_error(np.array(pred), np.array(true)),
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

        section_input_ids = d["section_input_ids"].to(device)
        section_attention_mask = d["section_attention_mask"].to(device)
        textlength = d["textlength"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        if MASK_WORDS == True: # masking some of the input ids (better way to do this?
            indices = np.random.choice(np.arange(len(attention_mask.flatten())), replace=False,
                                       size=int(len(attention_mask.flatten()) * 0.3))
            flat = attention_mask.flatten()
            flat[indices] = 0
            attention_mask = flat.reshape(attention_mask.shape)
        outputs = model(section_input_ids = section_input_ids,
                        section_attention_mask = section_attention_mask,
                        textlength=textlength)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        running_loss.append(loss.item())
        loss.backward()

        if batch_count % 45 == 0: #15 == 0: # update only every n batches (gradient accumulation) --> simulating bigger "batch size"
            #print(batch_count, "updating optimizer")
            #optimizer.step()
            #optimizer.zero_grad()

            optimizer_bert.step()
            optimizer_ffn.step()
            optimizer_weight_vector.step()
            optimizer_bert.zero_grad()
            optimizer_ffn.zero_grad()
            optimizer_weight_vector.zero_grad()

        if batch_count % 500 == 0: #300 == 0: # every 100 batches: write to tensorboard
            print(f"running train loss at batch {batch_count} (mean over last {len(running_loss)}):", np.mean(running_loss))
            # log the running train loss to tensorboard
            writer.add_scalar('train loss', np.mean(running_loss), batch_count)
            running_loss = []

        if batch_count % 1500 == 0: # every 300 batches: evaluate

            # EVALUATE
            eval_rt = evaluate_model(model = model)
            # log eval loss and pearson to tensorboard
            print("Mean eval loss:", eval_rt['eval_loss'])
            print("Pearson's r on dev set:", eval_rt['Pearson'])
            print("MSE on dev set:", eval_rt['MSE'])
            print("MAE on dev set:", eval_rt['MAE'])
            print("RAE on dev set:", eval_rt['RAE'])

            writer.add_scalar('eval loss', eval_rt['eval_loss'], batch_count)
            writer.add_scalar('Pearson', eval_rt['Pearson'], batch_count)
            writer.add_scalar('MSE', eval_rt['MSE'], batch_count)
            writer.add_scalar('MAE', eval_rt['MAE'], batch_count)
            writer.add_scalar('RAE', eval_rt['RAE'], batch_count)

            print("weight_vector values now:", model.weight_vector.data)

            model = model.train() # make sure it is back to train mode

    print("Mean train loss epoch:", np.mean(train_losses))
    writer.add_scalar('train loss epoch', np.mean(train_losses), epoch)

    print("saving model, optimizer, epoch, batch_count to", model_path)
    #torch.save(model.state_dict(), model_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_bert_state_dict': optimizer_bert.state_dict(),
        'optimizer_ffn_state_dict': optimizer_ffn.state_dict(),
        'optimizer_weight_vector': optimizer_weight_vector.state_dict(),
        'running_loss': running_loss,
        'batch_count': batch_count
                }, model_path)

# to load later
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
#
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# running_loss = checkpoint['running_loss']
# batch_count = checkpoint['batch_count']

#
# model.eval()
# # - or -
# model.train()

print("MAX_SECT: ", MAX_SECT)
print("SECTION_SIZE: ", SECTION_SIZE)
print("EPOCHS: ", EPOCHS)
print("BATCH SIZE: ", BATCH_SIZE)
print("LR: ", LR)

# nach 70 Epochen war: weight_vector values now: tensor([1.2078, 1.1993, 1.1906, 1.1719, 1.1554, 1.1367, 1.1184, 1.1134]

