import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from master_thesis.src import data, models, utils

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
#TODO: Vorsicht, das unterdrückt Warnungen

df = utils.get_conditioned_df(min_pageviews=100, max_pageviews=300)
#print(df.head())
df.reset_index(drop=True, inplace=True)  # so that index starts with 0 again

embs = utils.load_fasttext_vectors(limit=1000)
window = data.RandomWindow_CNN(start=None, fixed_len = None) # so kann man nach wie vor die ersten N Tokens nehmen
collater = data.Collater_CNN()

dl_train, dl_dev, dl_test = data.create_DataLoaders_CNN(df=df,
                                                        target = 'avgTimeOnPagePerNr_tokens',
                                                        text_base = 'text_preprocessed',
                                                        tokenizer = None,
                                                        train_batch_size = 10,
                                                        val_batch_size = 300,
                                                        transform=window,
                                                        collater=collater,
                                                        embs = embs)

data = next(iter(dl_train))
print(data.keys())
input_matrix = data['input_matrix']
print(input_matrix)
print(input_matrix.shape)
print(data['target'].shape)


"""
#assert torch.cuda.is_available()
#device = torch.device('cuda:0')
#print("Device is: ", device)
PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased' # 'distilbert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# try out tokenizer
sample_text = "Das hier ist ein deutscher Beispieltext. Und einen zweiten müssen wir auch noch haben."
tokens = tokenizer.tokenize(sample_text) # just tokenizes
token_ids = tokenizer.convert_tokens_to_ids(tokens)
ids = tokenizer.encode(sample_text) # already adds special tokens
encoded_plus = tokenizer.encode_plus(sample_text,
                                     max_length = None,
                                     return_token_type_ids=False,
                                     pad_to_max_length=False,
                                     truncation=False,
                                     return_attention_mask=True,)

print(tokens)
#print(token_ids)
#print(ids)
print("Testing the tokenizer:" , encoded_plus)

#tokenizer.get_vocab() # shows tokenizer vocab (subwords!)
#print(tokenizer.sep_token, tokenizer.sep_token_id, tokenizer.cls_token, tokenizer.cls_token_id, tokenizer.pad_token, tokenizer.pad_token_id)
#print(tokenizer.all_special_ids)
#print(tokenizer.all_special_tokens)

#window = data.RandomWindow(start=0, fixed_len = 200) # so kann man nach wie vor die ersten N Tokens nehmen

df = utils.get_conditioned_df(min_pageviews=100, max_pageviews=300)
#print(df.head())
df.reset_index(drop=True, inplace=True)  # so that index starts with 0 again
window = data.RandomWindow_BERT(start = None, fixed_len=None) # wenn man gerne eine fixed length haben will
collater = data.Collater_BERT()

dl_train, dl_dev, dl_test = data.create_DataLoaders_BERT(df=df,
                                                         target = 'avgTimeOnPagePerNr_tokens',
                                                         text_base = 'text_preprocessed',
                                                         tokenizer = tokenizer,
                                                         train_batch_size = 10,
                                                         val_batch_size = 300,
                                                         transform=window,
                                                         collater=collater)

data = next(iter(dl_train))
print(data.keys())
input_ids = data['input_ids']
print(input_ids)
print(input_ids.shape)
attention_mask = data['attention_mask']
#print(attention_mask)
print(attention_mask.shape)
print(data['target'].shape)



"""