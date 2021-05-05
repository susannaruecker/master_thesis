import torch
from torch import nn
from master_thesis.src import utils, models, data

from transformers import BertTokenizer
from torch.utils.data import DataLoader

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("device", help="specify which device to use ('cpu' or 'gpu')", type=str)
args = parser.parse_args()

device = torch.device('cpu' if args.device == 'cpu' else 'cuda')
print('Using device:', device)


# Loading a model with checkpoint

#model = models.BertFFN(n_outputs=1) # done
#identifier = 'BertFFN_FIXLEN512_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-03-03_00:13:33'

#model = models.BertAveraging(n_outputs=1) # done
#identifier = 'BertAveraging_FIXLEN512_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-03-31_16:45:24'

model = models.BertSequence(n_outputs=1) # TODO: hatte wohl den checkpoint gelöscht... trainiert neu
#identifier = 'BertSequence_FIXLEN512_MINLENNone_START0_EP30_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-05-04_19:04:19'
identifier = 'BertSequence_FIXLEN128_MINLENNone_START0_EP25_BS5_LR0.0001_avgTimeOnPage_NOZ' # todo: das ist mit 128 aber ist ja auch egal...

#model = models.BertTextlength(n_outputs = 1) # done
#identifier = 'BertTextlength_FIXLEN512_MINLENNone_START0_EP40_BS32_LR1e-05_avgTimeOnPage_NOZ_2021-04-27_11:37:43'


#model = models.BertHierarchical(n_outputs = 1, max_sect= 5) # done
#identifier = 'BertHierarchical_SECTIONSIZE400_MAX_SECT6_EP100_BS32_LR1e-05_avgTimeOnPage_NOZ_weighted_mean_2021-04-02_12:07:01'
#identifier = 'BertHierarchical_SECTIONSIZE512_MAX_SECT5_EP100_BS32_LR1e-05_avgTimeOnPage_NOZ_weighted_mean_2021-04-08_15:29:27'
#identifier = 'BertHierarchical_SECTIONSIZE512_MAX_SECT5_EP50_BS32_LR1e-05_avgTimeOnPage_NOZ_weighted_mean_pretrained_2021-04-19_12:21:17'
#SECTION_SIZE = 512
#MAX_SECT = 5

#model = models.BertHierarchicalRNN(n_outputs=1, max_sect= 5) # done
#identifier = 'BertHierarchicalRNN_SECTIONSIZE400_MAX_SECT6_EP50_BS32_LR1e-05_avgTimeOnPage_NOZ_GRU_2021-04-04_11:35:50'
#identifier = 'BertHierarchicalRNN_SECTIONSIZE512_MAX_SECT5_EP50_BS32_LR1e-05_avgTimeOnPage_NOZ_GRU_2021-04-14_11:14:43'
#identifier = 'BertHierarchicalRNN_SECTIONSIZE512_MAX_SECT5_EP50_BS32_LR1e-05_avgTimeOnPage_NOZ_GRU_pretrained_2021-04-20_23:07:20'
#SECTION_SIZE = 512
#MAX_SECT = 5

#EMBS_DIM = 300
#model = models.CNN(n_outputs=1, embs_dim=EMBS_DIM) # done
#identifier = 'CNN_FIXLEN800_MINLENNone_START0_EP50_BS32_LR0.0002_avgTimeOnPage_NOZ_2021-03-10_10:15:33'

#MODEL = "BertEmbs"
#model = models.EmbsFFN(n_outputs=1, input_size=768)
#identifier = 'BertEmbs_FIXLEN512_MINLENNone_START0_EP60_BS32_LR0.001_avgTimeOnPage_NOZ_2021-05-04_14:13:11'

#MODEL = "DANFastText"
#model = models.EmbsFFN(n_outputs=1, input_size=300)
#identifier = 'DANFastText_EP60_BS32_LR0.001_avgTimeOnPage_NOZ_2021-05-04_16:53:59'


model.to(device)

PATH = utils.OUTPUT / 'saved_models' / f'{identifier}'
model_state_dict = torch.load(PATH, map_location=torch.device('cpu'))["model_state_dict"]
model.load_state_dict(model_state_dict)

# tokenizer
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# DataSets and DataLoaders
# ACHTUNG: Je nach Model und Checkpoint anpassen!

GPU_BATCH = 32 # what can actually be done in one go on the GPU
FIXED_LEN = 512 #512 #128 #512
MIN_LEN = None # min window size (not used im FIXED_LEN is given)
START = 0 # random, if FIXED_LEN is specified you probably want to start at 0
PUBLISHER = 'NOZ'
TARGET = 'avgTimeOnPage'
FRACTION = 1

# for BertFFN (and similar) and BertTextlength
if model.__class__ in [models.BertTextlength, models.BertFFN, models.BertAveraging, models.BertSequence]:
    transform = data.TransformBERT(tokenizer = tokenizer, start = START, fixed_len = FIXED_LEN, min_len= MIN_LEN)
    collater = data.CollaterBERT()

# for BertHierarchical
if model.__class__ in [models.BertHierarchicalRNN, models.BertHierarchical]:
    GPU_BATCH = 1  # hier ist wichtig, nicht mehr als im Training zu nehmen, denn collater macht sonst Quatsch und padded...
                   # todo: das ist ein generelles Problem bei meinem Hierarchischen Bert...
    transform = data.TransformBERT(tokenizer = tokenizer, keep_all = True, start = None, fixed_len = None, min_len= None)
    collater = data.CollaterBERT_hierarchical(max_sect=MAX_SECT, section_size=SECTION_SIZE)

#for BertEmbs
if model.__class__ in [models.EmbsFFN] and MODEL == "BertEmbs":
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    FIXED_LEN = 512
    MIN_LEN = None
    START = 0
    transform = data.TransformBertFeatures(tokenizer=tokenizer, min_len=MIN_LEN, start=START,
                                           fixed_len=FIXED_LEN)  # document embedding: last hidden state of CLS-token
    collater = None

# for DAN
if model.__class__ in [models.EmbsFFN] and MODEL == "DANFastText":
    embs = utils.load_fasttext_vectors(limit=1000)  # now only dummy because using saved dataframe
    preprocessor = utils.Preprocessor(lemmatize=False,
                                      # ACHTUNG: in data.TransformDAN werden gespeicherte Daten benutzt, da ist Preprocessing vielleicht anders!
                                      delete_stopwords=True,
                                      delete_punctuation=True)
    transform = data.TransformDAN(embs=embs, preprocessor=preprocessor)  # document embedding: averaged word embeddings
                                                                         # ACHTUNG: das lädt bereits gespeicherte Feature, dahers ind Embs nur Dummy
    collater = None


# for CNN
if model.__class__ in [models.CNN]:
    embs = utils.load_fasttext_vectors(limit=None)
    FIXLEN = 800
    MINLEN = None
    START = 0
    transform = data.TransformCNN(tokenizer=None, embs=embs, start=START, fixed_len=FIXED_LEN, min_len=MIN_LEN)
    collater = data.CollaterCNN()

ds_dev = data.PublisherDataset(publisher=PUBLISHER, set = "dev", fraction=FRACTION, target = TARGET, text_base = "article_text", transform = transform)
ds_test = data.PublisherDataset(publisher=PUBLISHER, set = "test", fraction=FRACTION, target  = TARGET, text_base = "article_text", transform = transform)
print("Length of used DataSets:", len(ds_dev), len(ds_test))

dl_dev = DataLoader(ds_dev, batch_size=GPU_BATCH, num_workers=0, shuffle=False, collate_fn=collater, drop_last=False)
dl_test = DataLoader(ds_test, batch_size=GPU_BATCH, num_workers=0, shuffle=False, collate_fn=collater, drop_last=False)

loss_fn = nn.MSELoss()  # mean squared error

print("EVALUATING ON DEV AND TRAIN...")

print("------------ DEV SET ------------")
TESTING_ON = dl_dev
eval_dev = data.evaluate_model(model = model, dl = TESTING_ON, loss_fn=loss_fn,
                              using=args.device, max_batch=None)

print("Mean eval loss on DEV set:", eval_dev['eval_loss'])
print("Pearson's r on DEV set:", eval_dev['Pearson'])
print("Spearman's r on DEV set:", eval_dev['Spearman'])
print("MSE on DEV set:", eval_dev['MSE'])
print("MAE on DEV set:", eval_dev['MAE'])
print("RAE on DEV set:", eval_dev['RAE'])

df_dev = pd.DataFrame(0., index = eval_dev['articleIDs'], columns = ["true", "pred"])
df_dev["true"] = eval_dev["true"]
df_dev["pred"] = eval_dev["pred"]
print(df_dev.head())
df_dev.to_csv(utils.OUTPUT / "predictions" / "dev" / f'{identifier}.tsv', sep ="\t", index=True, index_label="articleId")

print("------------ TEST SET ------------")
TESTING_ON = dl_test
eval_test = data.evaluate_model(model = model, dl = TESTING_ON, loss_fn=loss_fn,
                              using=args.device, max_batch=None)

print("Mean eval loss on TEST set:", eval_test['eval_loss'])
print("Pearson's r on TEST set:", eval_test['Pearson'])
print("Spearman's r on TEST set:", eval_test['Spearman'])
print("MSE on TEST set:", eval_test['MSE'])
print("MAE on TEST set:", eval_test['MAE'])
print("RAE on TEST set:", eval_test['RAE'])

df_test = pd.DataFrame(0., index = eval_test['articleIDs'], columns = ["true", "pred"])
df_test["true"] = eval_test["true"]
df_test["pred"] = eval_test["pred"]
print(df_test.head())
df_test.to_csv(utils.OUTPUT / "predictions" / "test" / f'{identifier}.tsv', sep ="\t", index=True, index_label="articleId")
