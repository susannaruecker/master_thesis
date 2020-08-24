import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


class INWT_Dataset(Dataset):

    def __init__(self, df, target, text_base, tokenizer, max_len):
        self.df = df
        self.text_base = text_base
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.df.loc[item, self.text_base])
        target = np.array(self.df.loc[item, self.target])

        # TODO: hier einfach encode() nehmen? brauche ich die attention_mask etc?
        encoding = self.tokenizer.encode_plus(text,
                                              max_length=self.max_len,
                                              truncation=True,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        return {#'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'target': torch.tensor(target, dtype=torch.float).unsqueeze(dim=-1) # unsqueezing so shape (batch_size,1)
                }


def create_train_dev_test(df, random_seed = 123):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, shuffle=True)
    df_dev, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed, shuffle=True)
    df_train.reset_index(drop=True, inplace=True)  # so that index starts with 0 again
    df_dev.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    #print(df_train.shape, df_dev.shape, df_test.shape)
    return df_train, df_dev, df_test


def create_DataLoaders(df, target, text_base, tokenizer, max_len, batch_size):

    df_train, df_dev, df_test = create_train_dev_test(df = df, random_seed = 123)

    # creating DataSets
    ds_train = INWT_Dataset(df=df_train,
                            target=target,
                            text_base=text_base,
                            tokenizer=tokenizer,
                            max_len=max_len)
    ds_dev = INWT_Dataset(df=df_dev,
                          target=target,
                          text_base=text_base,
                          tokenizer=tokenizer,
                          max_len=max_len)
    ds_test = INWT_Dataset(df=df_test,
                           target=target,
                           text_base=text_base,
                           tokenizer=tokenizer,
                           max_len=max_len)

    # creating DataLoaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4, shuffle=True)
    dl_dev = DataLoader(ds_dev, batch_size=batch_size, num_workers=4)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=4)

    return dl_train, dl_dev, dl_test
