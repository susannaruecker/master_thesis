import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from master_thesis.src import utils, models
import json
import scipy.stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error


import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
#TODO: Vorsicht, das unterdrückt Warnungen


def create_train_dev_test(df, random_seed=123):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, shuffle=True)
    df_dev, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed, shuffle=True)

    train_IDs = df_train.index.tolist()
    dev_IDs = df_dev.index.tolist()
    test_IDs = df_test.index.tolist()

    splits = {'train': train_IDs, 'dev': dev_IDs, 'test': test_IDs}
    return splits


class PublisherDataset(Dataset):

    def __init__(self, publisher = "NOZ", set = "train", fraction = 1, target = "avgTimeOnPage", text_base = "article_text", transform=None):

        self.df = utils.get_publisher_df(publisher)
        with open(utils.OUTPUT / "splits" / f"{publisher}_splits.json", "r") as f:
            splits = json.load(f)
            set_IDs = splits[set]

        self.df = self.df.loc[set_IDs]
        self.df = self.df.sample(frac=fraction, replace=False, random_state=1) # possible for faster trials

        # try: min and max text length
        #self.df = self.df[self.df.nr_tokens_text >= 50]
        #self.df = self.df[self.df.nr_tokens_text <= 1500]
        #self.df = self.df[self.df.avgTimeOnPage <= 1800] # 30 Minuten

        print(f"Length of Dataset {set}", len(self.df))

        self.set = set
        self.text_base = text_base
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        articleId = self.df.iloc[item].name
        text = str(self.df.iloc[item][self.text_base])
        target = np.array(self.df.iloc[item][self.target])
        textlength = int(self.df.iloc[item].nr_tokens_text)
        publisher = utils.publisher_encoding[str(self.df.iloc[item].publisher)]

        rt = {
            'articleId': articleId,
            'text': text,
            'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1),  # unsqueezing so shape (batch_size,1)
            'textlength': torch.tensor(textlength, dtype=torch.float).unsqueeze(dim=-1),
            'publisher': torch.tensor(publisher, dtype = torch.long).unsqueeze(dim=-1)
            }

        if self.transform:
            rt = self.transform(rt)

        return rt


class TransformBERT(object):
    def __init__(self, tokenizer, keep_all= False, start=None, fixed_len=None, min_len = 200):
        self.fixed_len = fixed_len
        self.keep_all = keep_all
        self.start = start
        self.min_len = min_len
        self.tokenizer = tokenizer

    def __call__(self, sample):
        text = sample['text']

        encoding = self.tokenizer.encode_plus(text,
                                              max_length=None,
                                              truncation=False,
                                              return_token_type_ids=False,
                                              pad_to_max_length=False,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        original_len = len(input_ids)
        #print("original len:", original_len)
        if self.fixed_len:
            window_len = self.fixed_len
        elif self.min_len:
            window_len = np.random.randint(low=self.min_len, high=512)  # random window size between 200 and 510
        else:
            window_len = original_len

        if window_len > original_len:  # just in case text is shorter
            window_len = original_len  # take the original text length

        #print("window len", window_len)

        if self.start is not None:
            start = self.start
        elif original_len == window_len:
            start = 0
        else:
            start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
        end = start + window_len

        if self.keep_all == True: # ignore everything and take all!
            start = 0
            window_len = original_len

        end = start + window_len
        input_ids = input_ids[start:end]
        attention_mask = attention_mask[start:end]

        # make sure that special cls-token and end-token are there, even after transform (makes sense?)
        input_ids[0] = self.tokenizer.cls_token_id
        input_ids[-1] = self.tokenizer.sep_token_id
        attention_mask[0] = 1
        attention_mask[-1] = 1

        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        sample['BERT_tokens'] = torch.tensor(original_len, dtype=torch.float) # original textlength in BERT tokens

        return sample


class TransformDAN(object):
    def __init__(self, embs, preprocessor):
        self.embs = embs
        self.preprocessor = preprocessor
        self.fastText_Embs = pd.read_csv(utils.OUTPUT / 'Embs_features' / f'Embs_features_NOZ_full_nonlemmatized.tsv',
                       sep='\t', index_col="articleId") # already saved Embs for NOZ
        print("Using already precomputed feature matrix for NOZ!")


    def __call__(self, sample):
        # this would do them "from scratch"
        #text = sample['text']
        #doc_embedding = utils.get_averaged_vector(text = text, preprocessor = self.preprocessor, embs = self.embs)
        #sample['doc_embedding'] = torch.tensor(doc_embedding, dtype=torch.float)

        # quicker: look up embeddings (there already exists a df for NOZ)
        sample['doc_embedding'] = torch.tensor(self.fastText_Embs.loc[sample["articleId"]])

        return sample


class TransformBertFeatures(object):
    def __init__(self, tokenizer, keep_all= False, start=None, fixed_len=None, min_len = 200):
        self.fixed_len = fixed_len
        self.keep_all = keep_all
        self.start = start
        self.min_len = min_len
        self.tokenizer = tokenizer
        if fixed_len in [128, 512]: # already exists files with these
            print("using precomputed DF as feature lookup!")
            self.BERT_embs = pd.read_csv(utils.OUTPUT / 'BERT_features' / f'BERT_features_NOZ_full_FIXLEN{fixed_len}.tsv',
                           sep='\t', index_col="articleId") # todo: these are the ones with max_len = 128 or 512 !!!

    def __call__(self, sample):
        text = sample['text']

        encoding = self.tokenizer.encode_plus(text,
                                              max_length=None,
                                              truncation=False,
                                              return_token_type_ids=False,
                                              pad_to_max_length=False,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        original_len = len(input_ids)
        #print("original len:", original_len)
        if self.fixed_len:
            window_len = self.fixed_len
        elif self.min_len:
            window_len = np.random.randint(low=self.min_len, high=512)  # random window size between 200 and 510
        else:
            window_len = original_len

        if window_len > original_len:  # just in case text is shorter
            window_len = original_len  # take the original text length

        #print("window len", window_len)

        if self.start is not None:
            start = self.start
        elif original_len == window_len:
            start = 0
        else:
            start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
        end = start + window_len

        if self.keep_all == True: # ignore everything and take all!
            start = 0
            window_len = original_len

        end = start + window_len
        input_ids = input_ids[start:end]
        attention_mask = attention_mask[start:end]

        # make sure that special cls-token and end-token are there, even after transform (makes sense?)
        input_ids[0] = self.tokenizer.cls_token_id
        input_ids[-1] = self.tokenizer.sep_token_id
        attention_mask[0] = 1
        attention_mask[-1] = 1

        # get the already precomputed document embedding (hidden state of CLS-token):
        if self.fixed_len in [128, 512]:
            # quicker: look up embeddings (there already exists a df with the Bert Embeddings (max_len = 128 or 512):
            sample['doc_embedding'] = torch.tensor(self.BERT_embs.loc[sample["articleId"]])


        else:
            model = models.BERT_embedding()
            #device = torch.device("cuda")
            device = torch.device("cpu")

            model.to(device)
            model.eval()
            with torch.no_grad():
                doc_embedding = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            sample['doc_embedding'] = doc_embedding.squeeze()

        return sample


class TransformCNN(object):

    def __init__(self, tokenizer, embs, start=0, fixed_len=600, min_len=200):
        self.fixed_len = fixed_len
        self.start = start
        self.embs = embs
        self.tokenizer = tokenizer
        self.min_len = min_len

    def __call__(self, sample):
        text = sample['text']

        input_matrix = utils.get_embedding_matrix(text=text,
                                                  tokenizer=self.tokenizer,
                                                  embs=self.embs,
                                                  start = self.start,
                                                  fixed_length=self.fixed_len,
                                                  min_len = self.min_len)

        sample['input_matrix'] = torch.tensor(input_matrix, dtype=torch.float)

        return sample


class CollaterBERT():

    def __init__(self, padding_symbol = 0):
        self.padding_symbol = padding_symbol

    def __call__(self, samples, *args, **kwargs):

        batch = {}
        batch_input_ids = [ x['input_ids'].clone() for x in samples]
        batch_attention_mask = [ x['attention_mask'].clone() for x in samples]
        batch_target = [ x['target'].clone() for x in samples ]
        batch_textlength = [ x['textlength'].clone() for x in samples ]
        batch_BERT_tokens = [ x['BERT_tokens'].clone() for x in samples ]
        batch_publisher = [ x['publisher'].clone() for x in samples ]
        batch_articleId = [ x['articleId'] for x in samples ]


        lens = [len(x) for x in batch_input_ids]
        #print(lens)
        max_len = max(lens) # max_len = max(*lens)
        #print(max_len)

        tmp_input_ids = []
        tmp_attention_mask = []

        for i, input_ids in enumerate(batch_input_ids):
            curr_len = len(input_ids)
            padded_ids = torch.nn.functional.pad(input_ids, (0, max_len - curr_len), mode='constant', value= self.padding_symbol)
            tmp_input_ids.append(padded_ids)

        for i, attention_mask in enumerate(batch_attention_mask):
            curr_len = len(attention_mask)
            padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, max_len - curr_len), mode='constant', value=0)
            tmp_attention_mask.append(padded_attention_mask)

        batch_input_ids = torch.stack(tmp_input_ids)
        batch_attention_mask = torch.stack(tmp_attention_mask)

        batch['input_ids'] = batch_input_ids
        batch['attention_mask'] = batch_attention_mask
        batch['target'] = torch.tensor(batch_target, dtype=torch.float).unsqueeze(dim=-1)
        batch['textlength'] = torch.tensor(batch_textlength, dtype=torch.float).unsqueeze(dim=-1)
        batch['BERT_tokens'] = torch.tensor(batch_BERT_tokens, dtype=torch.float).unsqueeze(dim=-1)
        batch['publisher'] = torch.tensor(batch_publisher, dtype=torch.float).unsqueeze(dim=-1)
        batch['articleId'] = batch_articleId

        return batch


class CollaterCNN():

    def __init__(self, padding_symbol = 0):
        self.padding_symbol = padding_symbol

    def __call__(self, samples, *args, **kwargs):
        #print(samples)

        batch = {}
        batch_input_matrices = [ x['input_matrix'].clone() for x in samples]
        batch_target = [ x['target'].clone() for x in samples ]
        batch_textlength = [ x['textlength'].clone() for x in samples ]
        batch_publisher = [ x['publisher'].clone() for x in samples ]
        batch_articleId = [ x['articleId'] for x in samples ]

        lens = [len(x) for x in batch_input_matrices]
        #print(lens)
        max_len = max(lens) #max_len = max(*lens)
        #print(max_len)

        tmp_input_matrices = []

        for i, input_matrix in enumerate(batch_input_matrices):
            curr_len = len(input_matrix)
            padded_input_matrix = torch.nn.functional.pad(input_matrix,(0, 0, 0, (max_len-curr_len)), 'constant', 0)
            # das macht: padde nicht in der letzten Dimension (0,0,...)
            # padde in der vorletzten Dimension, aber da nur "hinten" (...,0,N)
            #print(padded_input_matrix)
            tmp_input_matrices.append(padded_input_matrix)

        batch_input_matrices = torch.stack(tmp_input_matrices)

        batch['input_matrix'] = batch_input_matrices
        batch['target'] = torch.tensor(batch_target, dtype=torch.float).unsqueeze(dim=-1)
        batch['textlength'] = torch.tensor(batch_textlength, dtype=torch.float).unsqueeze(dim=-1)
        batch['publisher'] = torch.tensor(batch_publisher, dtype=torch.float).unsqueeze(dim=-1)
        batch['articleId'] = batch_articleId

        return batch


class CollaterBERT_hierarchical():

    def __init__(self, padding_symbol = 0, section_size = 512, max_sect = 5):
        self.padding_symbol = padding_symbol
        self.section_size = section_size
        self.max_sect = max_sect

    def __call__(self, samples, *args, **kwargs):

        batch = {}
        batch_input_ids = [ x['input_ids'].clone() for x in samples]
        batch_attention_mask = [ x['attention_mask'].clone() for x in samples]
        batch_target = [ x['target'].clone() for x in samples ]
        batch_textlength = [ x['textlength'].clone() for x in samples ]
        batch_BERT_tokens = [ x['BERT_tokens'].clone() for x in samples ]
        batch_publisher = [ x['publisher'].clone() for x in samples ]
        batch_articleId = [ x['articleId'] for x in samples ]


        lens = [len(x) for x in batch_input_ids]
        max_len = max(lens) # max_len = max(*lens)
        max_sect = int(np.ceil(max_len/self.section_size)) # max number of section needed in this batch --> so size(batch_size,max_sect,512,786)
        if max_sect > self.max_sect:
            max_sect = self.max_sect
        needed_len = max_sect*self.section_size

        tmp_input_ids = []
        tmp_attention_mask = []

        for i, input_ids in enumerate(batch_input_ids):
            if needed_len <= len(input_ids):
                padded_ids = input_ids[:needed_len]
            else:
                padded_ids = torch.nn.functional.pad(input_ids, (0, needed_len - len(input_ids)),
                                                    mode='constant', value= self.padding_symbol)
            section_input_ids = torch.reshape(padded_ids, (max_sect, self.section_size))
            tmp_input_ids.append(section_input_ids)

        for i, attention_mask in enumerate(batch_attention_mask):
            if needed_len <= len(attention_mask):
                padded_attention_mask = attention_mask[:needed_len]
            else:
                padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, needed_len - len(attention_mask)),
                                                     mode='constant', value=self.padding_symbol)
            section_attention_mask = torch.reshape(padded_attention_mask, (max_sect, self.section_size))
            tmp_attention_mask.append(section_attention_mask)

        batch_section_input_ids = torch.stack(tmp_input_ids)
        batch_section_attention_mask = torch.stack(tmp_attention_mask)

        batch['section_input_ids'] = batch_section_input_ids
        batch['section_attention_mask'] = batch_section_attention_mask
        batch['target'] = torch.tensor(batch_target, dtype=torch.float).unsqueeze(dim=-1)
        batch['textlength'] = torch.tensor(batch_textlength, dtype=torch.float).unsqueeze(dim=-1)
        batch['BERT_tokens'] = torch.tensor(batch_BERT_tokens, dtype=torch.float).unsqueeze(dim=-1)
        batch['publisher'] = torch.tensor(batch_publisher, dtype=torch.float).unsqueeze(dim=-1)
        batch['articleId'] = batch_articleId

        return batch


#### for evaluating the DL models during training ###
# todo: write the values in some kind of log?
def evaluate_model(model, dl, loss_fn, using = "cpu", max_batch = None):
    if using == "cpu":
        device = torch.device('cpu')
    if using == "gpu":
        device = torch.device('cuda')
    print('Using device:', device)
    model.eval()
    eval_losses = []

    pred = []  # for calculating Pearson's r on dev
    true = []
    articleIDs = []

    with torch.no_grad():
        for nr, d in enumerate(dl):
            print("-Batch", nr, end='\r')
            targets = d["target"].to(device)
            IDs = d["articleId"]
            if model.__class__ in [models.BertTextlength]:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                #textlength = d["BERT_tokens"].to(device)
                textlength = d["textlength"].to(device) # TODO
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, textlength = textlength)
            elif model.__class__ in [models.BertHierarchical, models.BertHierarchicalRNN]:
                #textlength = d["BERT_tokens"].to(device)
                section_input_ids = d["section_input_ids"].to(device)
                section_attention_mask = d["section_attention_mask"].to(device)
                outputs = model(section_input_ids=section_input_ids,
                                section_attention_mask=section_attention_mask)
                                #textlength=textlength)
            elif model.__class__ in [models.baseline_textlength]:
                textlength = d["BERT_tokens"].to(device)
                outputs = model(textlength = textlength)
            elif model.__class__ in [models.BertFFN, models.BertSequence, models.BertAveraging]:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            elif model.__class__ in [models.EmbsFFN]:
                vector = d["doc_embedding"].to(device)
                outputs = model(vector = vector)
            elif model.__class__ in [models.CNN]:
                input_matrix = d["input_matrix"].to(device)
                outputs = model(input_matrix = input_matrix)
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())

            outputs = outputs.squeeze().cpu()
            targets = targets.squeeze().cpu()

            if len(d["target"]) == 1:  # necessary if BATCH_SIZE = 1
                pred.append(outputs)
                true.append(targets)
                articleIDs.append(IDs[0]) # stupid, but IDs is a list with one item...
            else:
                pred.extend(outputs)
                true.extend(targets)
                articleIDs.extend(IDs)

            if max_batch:
                if nr >= max_batch:
                    break

    print("Inspecting some predicted and their true values:")
    print("- pred:", [round(t.item(), 2) for t in pred[:10]])
    print("- true:", [round(t.item(), 2) for t in true[:10]])
    print("- MEAN and STD of predicted values:", np.mean(pred), np.std(pred))
    print("- MEAN and STD of true values:", np.mean(true), np.std(true))

    print("- Length of dataset used to evaluate:", len(pred))
    pred_list = [ round(t.item(), 4) for t in pred ]
    true_list = [ round(t.item(), 4) for t in true ]


    return {'Pearson': st.pearsonr(pred, true)[0],
            'Spearman': st.spearmanr(pred, true)[0],
            'MSE': mean_squared_error(pred, true),
            'MAE': mean_absolute_error(pred, true),
            'RAE': utils.relative_absolute_error(np.array(pred), np.array(true)),
            'eval_loss': np.mean(eval_losses),
            'pred': pred_list,
            'true': true_list,
            'articleIDs': articleIDs}






#####
#####
##### OLDER AND (hopefully) DEPRECATED
#####
#####

# class INWT_Dataset_BERT(Dataset):
#
#     def __init__(self, df, target, text_base, tokenizer, transform=None):
#         self.df = df
#         self.text_base = text_base
#         self.target = target
#         self.tokenizer = tokenizer
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, item):
#         text = str(self.df.loc[item, self.text_base])
#         target = np.array(self.df.loc[item, self.target])
#
#         # TODO: hier einfach encode() nehmen? brauche ich die attention_mask etc?
#         encoding = self.tokenizer.encode_plus(text,
#                                               max_length=None,
#                                               truncation=False,
#                                               return_token_type_ids=False,
#                                               pad_to_max_length=False,
#                                               return_attention_mask=True,
#                                               return_tensors='pt',
#                                               )
#
#         rt = {  # 'text': text,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1)  # unsqueezing so shape (batch_size,1)
#         }
#         if self.transform:
#             rt = self.transform(rt)
#             # make sure that special cls-token and end-token are there, even after transform (makes sense?)
#             rt['input_ids'][0] = self.tokenizer.cls_token_id
#             rt['input_ids'][-1] = self.tokenizer.sep_token_id
#             rt['attention_mask'][0] = 1
#             rt['attention_mask'][-1] = 1
#
#         return rt
#
#
# class RandomWindow_BERT(object):
#     """
#     returns a window from the input_ids
#     :param start: start can be specified (e.g. 0 for beginning)
#     :param fixed_len: can be specified (e.g. 200 for window of size 200)
#     """
#
#     def __init__(self, start=None, fixed_len=None, min_len = 200):
#         self.fixed_len = fixed_len
#         self.start = start
#         self.min_len = min_len
#
#     def __call__(self, sample):
#         input_ids, attention_mask, target = sample['input_ids'], sample['attention_mask'], sample['target']
#
#         original_len = len(input_ids)
#         #print("original len", original_len)
#         if self.fixed_len:
#             window_len = self.fixed_len
#         else:
#             window_len = np.random.randint(low=self.min_len, high=512)  # random window size between 200 and 510
#
#         if window_len > original_len:  # just in case text is shorter
#             window_len = original_len  # take the original text length
#
#         #print("window len", window_len)
#
#         if self.start is not None:
#             start = self.start
#         elif original_len == window_len:
#             start = 0
#         else:
#             start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
#         #print("start", start)
#         end = start + window_len
#
#         input_ids = input_ids[start:end]
#         attention_mask = attention_mask[start:end]
#         target = target
#
#         return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target': target}
#
#
# class Collater_BERT():
#
#     def __init__(self, padding_symbol = 0):
#         self.padding_symbol = padding_symbol
#
#     def __call__(self, samples, *args, **kwargs):
#         #print(samples)
#
#         batch = {}
#         batch_input_ids = [ x['input_ids'].clone() for x in samples]
#         batch_attention_mask = [ x['attention_mask'].clone() for x in samples]
#         batch_target = [ x['target'].clone() for x in samples ]
#
#         lens = [len(x) for x in batch_input_ids]
#         #print(lens)
#         max_len = max(lens) # max_len = max(*lens)
#         #print(max_len)
#
#         tmp_input_ids = []
#         tmp_attention_mask = []
#
#         for i, input_ids in enumerate(batch_input_ids):
#             curr_len = len(input_ids)
#             padded_ids = torch.nn.functional.pad(input_ids, (0, max_len - curr_len), mode='constant', value= self.padding_symbol)
#             tmp_input_ids.append(padded_ids)
#
#         for i, attention_mask in enumerate(batch_attention_mask):
#             curr_len = len(attention_mask)
#             padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, max_len - curr_len), mode='constant', value=0)
#             tmp_attention_mask.append(padded_attention_mask)
#
#         batch_input_ids = torch.stack(tmp_input_ids)
#         batch_attention_mask = torch.stack(tmp_attention_mask)
#
#         batch['input_ids'] = batch_input_ids
#         batch['attention_mask'] = batch_attention_mask
#         batch['target'] = torch.tensor(batch_target, dtype=torch.float).unsqueeze(dim=-1)
#
#         return batch
#
#
#
#
#
# class INWT_Dataset_CNN(Dataset):
#
#     def __init__(self, df, target, text_base, tokenizer, embs, max_len=None, transform=None):
#         self.df = df
#         self.text_base = text_base
#         self.target = target
#         self.tokenizer = tokenizer
#         self.embs = embs
#         self.transform = transform
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, item):
#         text = str(self.df.loc[item, self.text_base])
#         target = np.array(self.df.loc[item, self.target])
#         input_matrix = utils.get_embedding_matrix(text=text,
#                                                   tokenizer=self.tokenizer,
#                                                   embs=self.embs,
#                                                   fixed_length=self.max_len)
#
#         rt = {  # 'text': text,
#             'input_matrix': torch.tensor(input_matrix, dtype=torch.float),
#             'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1)  # unsqueezing so shape (batch_size,1)
#             }
#
#         if self.transform:
#             rt = self.transform(rt)
#
#         return rt
#
#
# class RandomWindow_CNN(object):
#     """
#     returns a window from the input_ids
#     :param start: start can be specified (e.g. 0 for beginning)
#     :param fixed_len: can be specified (e.g. 300 for window of size 200)
#     """
#
#     def __init__(self, start=None, fixed_len=None, min_len = 200):
#         self.fixed_len = fixed_len
#         self.start = start
#         self.min_len = min_len
#
#     def __call__(self, sample):
#         input_matrix, target = sample['input_matrix'], sample['target']
#
#         original_len = len(input_matrix)
#         #print("original len", original_len)
#         if self.fixed_len:
#             window_len = self.fixed_len
#         else:
#             window_len = np.random.randint(low=self.min_len, high=800)  # random window size between 200 and 800
#
#         if window_len > original_len:  # just in case text is shorter
#             window_len = original_len  # take the original text length
#
#         #print("window len", window_len)
#
#         if self.start is not None:
#             start = self.start
#         elif original_len == window_len:
#             start = 0
#         else:
#             start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
#         #print("start", start)
#         end = start + window_len
#
#         input_matrix = input_matrix[start:end]
#         target = target
#
#         return {'input_matrix': input_matrix, 'target': target}
#
# class Collater_CNN():
#
#     def __init__(self, padding_symbol = 0):
#         self.padding_symbol = padding_symbol
#
#     def __call__(self, samples, *args, **kwargs):
#         #print(samples)
#
#         batch = {}
#         batch_input_matrices = [ x['input_matrix'].clone() for x in samples]
#         batch_target = [ x['target'].clone() for x in samples ]
#
#         lens = [len(x) for x in batch_input_matrices]
#         #print(lens)
#         max_len = max(lens) #max_len = max(*lens)
#         #print(max_len)
#
#         tmp_input_matrices = []
#
#         for i, input_matrix in enumerate(batch_input_matrices):
#             curr_len = len(input_matrix)
#             padded_input_matrix = torch.nn.functional.pad(input_matrix,(0, 0, 0, (max_len-curr_len)), 'constant', 0)
#             # das macht: padde nicht in der letzten Dimension (0,0,...)
#             # padde in der vorletzten Dimension, aber da nur "hinten" (...,0,N)
#             #print(padded_input_matrix)
#             tmp_input_matrices.append(padded_input_matrix)
#
#         batch_input_matrices = torch.stack(tmp_input_matrices)
#
#         batch['input_matrix'] = batch_input_matrices
#         batch['target'] = torch.tensor(batch_target, dtype=torch.float).unsqueeze(dim=-1)
#
#         return batch
#
# class INWT_Dataset_FFN_BERT(Dataset):
#
#     def __init__(self, df, target, text_base, tokenizer, transform=None):
#         self.df = df
#         self.text_base = text_base
#         self.target = target
#         self.tokenizer = tokenizer
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, item):
#         text = str(self.df.loc[item, self.text_base])
#         target = np.array(self.df.loc[item, self.target])
#         nr_tokens = int(self.df.loc[item, 'nr_tokens_publisher'])
#         publisher = utils.publisher_encoding[str(self.df.loc[item, 'publisher'])]
#
#         encoding = self.tokenizer.encode_plus(text,
#                                               max_length=None,
#                                               truncation=False,
#                                               return_token_type_ids=False,
#                                               pad_to_max_length=False,
#                                               return_attention_mask=True,
#                                               return_tensors='pt',
#                                               )
#
#         rt = {  # 'text': text,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1),  # unsqueezing so shape (batch_size,1)
#             'textlength': torch.tensor(nr_tokens, dtype=torch.float).unsqueeze(dim=-1),
#             'publisher': torch.tensor(publisher, dtype = torch.long).unsqueeze(dim=-1)
#         }
#         if self.transform:
#             rt = self.transform(rt)
#             # make sure that special cls-token and end-token are there, even after transform (makes sense?)
#             rt['input_ids'][0] = self.tokenizer.cls_token_id
#             rt['input_ids'][-1] = self.tokenizer.sep_token_id
#             rt['attention_mask'][0] = 1
#             rt['attention_mask'][-1] = 1
#
#         return rt
#
#
# class INWT_Dataset_baseline(Dataset):
#
#     def __init__(self, df, target, text_base):
#         self.df = df
#         self.text_base = text_base
#         self.target = target
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, item):
#         text = str(self.df.loc[item, self.text_base])
#         target = np.array(self.df.loc[item, self.target])
#         nr_tokens = int(self.df.loc[item, 'nr_tokens_publisher'])
#         publisher = utils.publisher_encoding[str(self.df.loc[item, 'publisher'])]
#
#         rt = {  # 'text': text,
#             'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1),  # unsqueezing so shape (batch_size,1)
#             'textlength': torch.tensor(nr_tokens, dtype=torch.float).unsqueeze(dim=-1),
#             'publisher': torch.tensor(publisher, dtype = torch.long).unsqueeze(dim=-1)
#         }
#
#         return rt
#
#
#
# class RandomWindow_FFN_BERT(object):
#     """
#     returns a window from the input_ids
#     :param start: start can be specified (e.g. 0 for beginning)
#     :param fixed_len: can be specified (e.g. 200 for window of size 200)
#     """
#
#     def __init__(self, start=None, fixed_len=None, min_len = 200):
#         self.fixed_len = fixed_len
#         self.start = start
#         self.min_len = min_len
#
#     def __call__(self, sample):
#         input_ids, attention_mask, target = sample['input_ids'], sample['attention_mask'], sample['target']
#         textlength, publisher = sample['textlength'], sample['publisher']
#
#         original_len = len(input_ids)
#         #print("original len", original_len)
#         if self.fixed_len:
#             window_len = self.fixed_len
#         else:
#             window_len = np.random.randint(low=self.min_len, high=512)  # random window size between 200 and 510
#
#         if window_len > original_len:  # just in case text is shorter
#             window_len = original_len  # take the original text length
#
#         #print("window len", window_len)
#
#         if self.start is not None:
#             start = self.start
#         elif original_len == window_len:
#             start = 0
#         else:
#             start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
#         #print("start", start)
#         end = start + window_len
#
#         input_ids = input_ids[start:end]
#         attention_mask = attention_mask[start:end]
#         target = target
#
#
#         return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target': target,
#                 'textlength': textlength, 'publisher': publisher}
#
#
# class Collater_FFN_BERT():
#
#     def __init__(self, padding_symbol = 0):
#         self.padding_symbol = padding_symbol
#
#     def __call__(self, samples, *args, **kwargs):
#         #print(samples)
#
#         batch = {}
#         batch_input_ids = [ x['input_ids'].clone() for x in samples]
#         batch_attention_mask = [ x['attention_mask'].clone() for x in samples]
#         batch_target = [ x['target'].clone() for x in samples ]
#         batch_textlength = [ x['textlength'].clone() for x in samples ]
#         batch_publisher = [ x['publisher'].clone() for x in samples ]
#
#         lens = [len(x) for x in batch_input_ids]
#         #print(lens)
#         max_len = max(lens) # max_len = max(*lens)
#         #print(max_len)
#
#         tmp_input_ids = []
#         tmp_attention_mask = []
#
#         for i, input_ids in enumerate(batch_input_ids):
#             curr_len = len(input_ids)
#             padded_ids = torch.nn.functional.pad(input_ids, (0, max_len - curr_len), mode='constant', value= self.padding_symbol)
#             tmp_input_ids.append(padded_ids)
#
#         for i, attention_mask in enumerate(batch_attention_mask):
#             curr_len = len(attention_mask)
#             padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, max_len - curr_len), mode='constant', value=0)
#             tmp_attention_mask.append(padded_attention_mask)
#
#         batch_input_ids = torch.stack(tmp_input_ids)
#         batch_attention_mask = torch.stack(tmp_attention_mask)
#
#         batch['input_ids'] = batch_input_ids
#         batch['attention_mask'] = batch_attention_mask
#         batch['target'] = torch.tensor(batch_target, dtype=torch.float).unsqueeze(dim=-1)
#         batch['textlength'] = torch.tensor(batch_textlength, dtype=torch.float).unsqueeze(dim=-1)
#         batch['publisher'] = torch.tensor(batch_publisher, dtype=torch.long).unsqueeze(dim=-1)
#
#         return batch





# def create_DataLoaders_BERT(df, target, text_base, tokenizer, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
#     df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)
#
#     # creating DataSets
#     ds_train = INWT_Dataset_BERT(df=df_train,
#                                  target=target,
#                                  text_base=text_base,
#                                  tokenizer=tokenizer,
#                                  transform=transform)
#     ds_dev = INWT_Dataset_BERT(df=df_dev,
#                                target=target,
#                                text_base=text_base,
#                                tokenizer=tokenizer,
#                                transform=transform)
#     ds_test = INWT_Dataset_BERT(df=df_test,
#                                 target=target,
#                                 text_base=text_base,
#                                 tokenizer=tokenizer,
#                                 transform=transform)
#
#     # creating DataLoaders
#     dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#
#     return dl_train, dl_dev, dl_test


# def create_DataLoaders_CNN(df, target, text_base, tokenizer, embs, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
#     df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)
#
#     # creating DataSets
#     ds_train = INWT_Dataset_CNN(df=df_train,
#                                 target=target,
#                                 text_base=text_base,
#                                 tokenizer=tokenizer,
#                                 embs=embs,
#                                 transform=transform)
#     ds_dev = INWT_Dataset_CNN(df=df_dev,
#                               target=target,
#                               text_base=text_base,
#                               tokenizer=tokenizer,
#                               embs=embs,
#                               transform=transform)
#     ds_test = INWT_Dataset_CNN(df=df_test,
#                                target=target,
#                                text_base=text_base,
#                                tokenizer=tokenizer,
#                                embs=embs,
#                                transform=transform)
#
#     # creating DataLoaders
#     dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, collate_fn=collater)
#     dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, collate_fn=collater)
#
#    return dl_train, dl_dev, dl_test


# def create_DataLoaders_FFN_BERT(df, target, text_base, tokenizer, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
#     df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)
#
#     # creating DataSets
#     ds_train = INWT_Dataset_FFN_BERT(df=df_train,
#                                  target=target,
#                                  text_base=text_base,
#                                  tokenizer=tokenizer,
#                                  transform=transform)
#     ds_dev = INWT_Dataset_FFN_BERT(df=df_dev,
#                                target=target,
#                                text_base=text_base,
#                                tokenizer=tokenizer,
#                                transform=transform)
#     ds_test = INWT_Dataset_FFN_BERT(df=df_test,
#                                 target=target,
#                                 text_base=text_base,
#                                 tokenizer=tokenizer,
#                                 transform=transform)
#
#     # creating DataLoaders
#     dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#
#     return dl_train, dl_dev, dl_test

# def create_DataLoaders_baseline(df, target, text_base, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
#     df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)
#
#     # creating DataSets
#     ds_train = INWT_Dataset_baseline(df=df_train,
#                                  target=target,
#                                  text_base=text_base)
#     ds_dev = INWT_Dataset_baseline(df=df_dev,
#                                target=target,
#                                text_base=text_base)
#     ds_test = INWT_Dataset_baseline(df=df_test,
#                                 target=target,
#                                 text_base=text_base)
#
#     # creating DataLoaders
#     dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#     dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
#
#     return dl_train, dl_dev, dl_test
