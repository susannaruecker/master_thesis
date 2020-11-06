import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from master_thesis.src import utils

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
#TODO: Vorsicht, das unterdrückt Warnungen

class INWT_Dataset_BERT(Dataset):

    def __init__(self, df, target, text_base, tokenizer, transform=None):
        self.df = df
        self.text_base = text_base
        self.target = target
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.df.loc[item, self.text_base])
        target = np.array(self.df.loc[item, self.target])

        # TODO: hier einfach encode() nehmen? brauche ich die attention_mask etc?
        encoding = self.tokenizer.encode_plus(text,
                                              max_length=None,
                                              truncation=False,
                                              return_token_type_ids=False,
                                              pad_to_max_length=False,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        rt = {  # 'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1)  # unsqueezing so shape (batch_size,1)
        }
        if self.transform:
            rt = self.transform(rt)
            # make sure that special cls-token and end-token are there, even after transform (makes sense?)
            rt['input_ids'][0] = self.tokenizer.cls_token_id
            rt['input_ids'][-1] = self.tokenizer.sep_token_id
            rt['attention_mask'][0] = 1
            rt['attention_mask'][-1] = 1

        return rt


class RandomWindow_BERT(object):
    """
    returns a window from the input_ids
    :param start: start can be specified (e.g. 0 for beginning)
    :param fixed_len: can be specified (e.g. 200 for window of size 200)
    """

    def __init__(self, start=None, fixed_len=None, min_len = 200):
        self.fixed_len = fixed_len
        self.start = start
        self.min_len = min_len

    def __call__(self, sample):
        input_ids, attention_mask, target = sample['input_ids'], sample['attention_mask'], sample['target']

        original_len = len(input_ids)
        #print("original len", original_len)
        if self.fixed_len:
            window_len = self.fixed_len
        else:
            window_len = np.random.randint(low=self.min_len, high=512)  # random window size between 200 and 510

        if window_len > original_len:  # just in case text is shorter
            window_len = original_len  # take the original text length

        #print("window len", window_len)

        if self.start is not None:
            start = self.start
        elif original_len == window_len:
            start = 0
        else:
            start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
        #print("start", start)
        end = start + window_len

        input_ids = input_ids[start:end]
        attention_mask = attention_mask[start:end]
        target = target

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target': target}


class Collater_BERT():

    def __init__(self, padding_symbol = 0):
        self.padding_symbol = padding_symbol

    def __call__(self, samples, *args, **kwargs):
        #print(samples)

        batch = {}
        batch_input_ids = [ x['input_ids'].clone() for x in samples]
        batch_attention_mask = [ x['attention_mask'].clone() for x in samples]
        batch_target = [ x['target'].clone() for x in samples ]

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

        return batch





class INWT_Dataset_CNN(Dataset):

    def __init__(self, df, target, text_base, tokenizer, embs, max_len=None, transform=None):
        self.df = df
        self.text_base = text_base
        self.target = target
        self.tokenizer = tokenizer
        self.embs = embs
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.df.loc[item, self.text_base])
        target = np.array(self.df.loc[item, self.target])
        input_matrix = utils.get_embedding_matrix(text=text,
                                                  tokenizer=self.tokenizer,
                                                  embs=self.embs,
                                                  fixed_length=self.max_len)

        rt = {  # 'text': text,
            'input_matrix': torch.tensor(input_matrix, dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1)  # unsqueezing so shape (batch_size,1)
            }

        if self.transform:
            rt = self.transform(rt)

        return rt


class RandomWindow_CNN(object):
    """
    returns a window from the input_ids
    :param start: start can be specified (e.g. 0 for beginning)
    :param fixed_len: can be specified (e.g. 300 for window of size 200)
    """

    def __init__(self, start=None, fixed_len=None, min_len = 200):
        self.fixed_len = fixed_len
        self.start = start
        self.min_len = min_len

    def __call__(self, sample):
        input_matrix, target = sample['input_matrix'], sample['target']

        original_len = len(input_matrix)
        #print("original len", original_len)
        if self.fixed_len:
            window_len = self.fixed_len
        else:
            window_len = np.random.randint(low=self.min_len, high=800)  # random window size between 200 and 800

        if window_len > original_len:  # just in case text is shorter
            window_len = original_len  # take the original text length

        #print("window len", window_len)

        if self.start is not None:
            start = self.start
        elif original_len == window_len:
            start = 0
        else:
            start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
        #print("start", start)
        end = start + window_len

        input_matrix = input_matrix[start:end]
        target = target

        return {'input_matrix': input_matrix, 'target': target}

class Collater_CNN():

    def __init__(self, padding_symbol = 0):
        self.padding_symbol = padding_symbol

    def __call__(self, samples, *args, **kwargs):
        #print(samples)

        batch = {}
        batch_input_matrices = [ x['input_matrix'].clone() for x in samples]
        batch_target = [ x['target'].clone() for x in samples ]

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

        return batch

class INWT_Dataset_FFN_BERT(Dataset):

    def __init__(self, df, target, text_base, tokenizer, transform=None):
        self.df = df
        self.text_base = text_base
        self.target = target
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.df.loc[item, self.text_base])
        target = np.array(self.df.loc[item, self.target])
        nr_tokens = int(self.df.loc[item, 'nr_tokens_publisher'])
        publisher = utils.publisher_encoding[str(self.df.loc[item, 'publisher'])]

        encoding = self.tokenizer.encode_plus(text,
                                              max_length=None,
                                              truncation=False,
                                              return_token_type_ids=False,
                                              pad_to_max_length=False,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        rt = {  # 'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1),  # unsqueezing so shape (batch_size,1)
            'textlength': torch.tensor(nr_tokens, dtype=torch.float).unsqueeze(dim=-1),
            'publisher': torch.tensor(publisher, dtype = torch.long).unsqueeze(dim=-1)
        }
        if self.transform:
            rt = self.transform(rt)
            # make sure that special cls-token and end-token are there, even after transform (makes sense?)
            rt['input_ids'][0] = self.tokenizer.cls_token_id
            rt['input_ids'][-1] = self.tokenizer.sep_token_id
            rt['attention_mask'][0] = 1
            rt['attention_mask'][-1] = 1

        return rt


class RandomWindow_FFN_BERT(object):
    """
    returns a window from the input_ids
    :param start: start can be specified (e.g. 0 for beginning)
    :param fixed_len: can be specified (e.g. 200 for window of size 200)
    """

    def __init__(self, start=None, fixed_len=None, min_len = 200):
        self.fixed_len = fixed_len
        self.start = start
        self.min_len = min_len

    def __call__(self, sample):
        input_ids, attention_mask, target = sample['input_ids'], sample['attention_mask'], sample['target']
        textlength, publisher = sample['textlength'], sample['publisher']

        original_len = len(input_ids)
        #print("original len", original_len)
        if self.fixed_len:
            window_len = self.fixed_len
        else:
            window_len = np.random.randint(low=self.min_len, high=512)  # random window size between 200 and 510

        if window_len > original_len:  # just in case text is shorter
            window_len = original_len  # take the original text length

        #print("window len", window_len)

        if self.start is not None:
            start = self.start
        elif original_len == window_len:
            start = 0
        else:
            start = np.random.randint(low=0, high=original_len - window_len)  # start shouldn't be too late
        #print("start", start)
        end = start + window_len

        input_ids = input_ids[start:end]
        attention_mask = attention_mask[start:end]
        target = target

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target': target,
                'textlength': textlength, 'publisher': publisher}


class Collater_FFN_BERT():

    def __init__(self, padding_symbol = 0):
        self.padding_symbol = padding_symbol

    def __call__(self, samples, *args, **kwargs):
        #print(samples)

        batch = {}
        batch_input_ids = [ x['input_ids'].clone() for x in samples]
        batch_attention_mask = [ x['attention_mask'].clone() for x in samples]
        batch_target = [ x['target'].clone() for x in samples ]
        batch_textlength = [ x['textlength'].clone() for x in samples ]
        batch_publisher = [ x['publisher'].clone() for x in samples ]

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
        batch['publisher'] = torch.tensor(batch_publisher, dtype=torch.long).unsqueeze(dim=-1)

        return batch



def create_train_dev_test(df, random_seed=123):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, shuffle=True)
    df_dev, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed, shuffle=True)
    df_train.reset_index(drop=True, inplace=True)  # so that index starts with 0 again
    df_dev.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    # print(df_train.shape, df_dev.shape, df_test.shape)
    return df_train, df_dev, df_test


def create_DataLoaders_BERT(df, target, text_base, tokenizer, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
    df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)

    # creating DataSets
    ds_train = INWT_Dataset_BERT(df=df_train,
                                 target=target,
                                 text_base=text_base,
                                 tokenizer=tokenizer,
                                 transform=transform)
    ds_dev = INWT_Dataset_BERT(df=df_dev,
                               target=target,
                               text_base=text_base,
                               tokenizer=tokenizer,
                               transform=transform)
    ds_test = INWT_Dataset_BERT(df=df_test,
                                target=target,
                                text_base=text_base,
                                tokenizer=tokenizer,
                                transform=transform)

    # creating DataLoaders
    dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
    dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
    dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)

    return dl_train, dl_dev, dl_test


def create_DataLoaders_CNN(df, target, text_base, tokenizer, embs, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
    df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)

    # creating DataSets
    ds_train = INWT_Dataset_CNN(df=df_train,
                                target=target,
                                text_base=text_base,
                                tokenizer=tokenizer,
                                embs=embs,
                                transform=transform)
    ds_dev = INWT_Dataset_CNN(df=df_dev,
                              target=target,
                              text_base=text_base,
                              tokenizer=tokenizer,
                              embs=embs,
                              transform=transform)
    ds_test = INWT_Dataset_CNN(df=df_test,
                               target=target,
                               text_base=text_base,
                               tokenizer=tokenizer,
                               embs=embs,
                               transform=transform)

    # creating DataLoaders
    dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
    dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, collate_fn=collater)
    dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, collate_fn=collater)

    return dl_train, dl_dev, dl_test


def create_DataLoaders_FFN_BERT(df, target, text_base, tokenizer, train_batch_size=12, val_batch_size = 12, transform=None, collater=None):
    df_train, df_dev, df_test = create_train_dev_test(df=df, random_seed=123)

    # creating DataSets
    ds_train = INWT_Dataset_FFN_BERT(df=df_train,
                                 target=target,
                                 text_base=text_base,
                                 tokenizer=tokenizer,
                                 transform=transform)
    ds_dev = INWT_Dataset_FFN_BERT(df=df_dev,
                               target=target,
                               text_base=text_base,
                               tokenizer=tokenizer,
                               transform=transform)
    ds_test = INWT_Dataset_FFN_BERT(df=df_test,
                                target=target,
                                text_base=text_base,
                                tokenizer=tokenizer,
                                transform=transform)

    # creating DataLoaders
    dl_train = DataLoader(ds_train, batch_size=train_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
    dl_dev = DataLoader(ds_dev, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)
    dl_test = DataLoader(ds_test, batch_size=val_batch_size, num_workers=4, shuffle=True, collate_fn=collater)

    return dl_train, dl_dev, dl_test

