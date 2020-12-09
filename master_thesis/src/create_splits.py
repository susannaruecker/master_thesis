from master_thesis.src import models, data, utils
from transformers import BertTokenizer, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import json

# #####
#
# NOZ = utils.get_publisher_df("NOZ")
# print(NOZ.head())
#
# splits = data.create_train_dev_test(df = NOZ, random_seed=123)
# with open(utils.OUTPUT / "splits" / "NOZ_splits.json", "w") as f:
#     json.dump(splits, f)
#
# #####
#
# SZ = utils.get_publisher_df("SZ")
# print(SZ.head())
#
# splits = data.create_train_dev_test(df = SZ, random_seed=123)
# with open(utils.OUTPUT / "splits"/ "SZ_splits.json", "w") as f:
#     json.dump(splits, f)
#
# #####
#
# TV = utils.get_publisher_df("TV")
# print(TV.head())
#
# splits = data.create_train_dev_test(df = TV, random_seed=123)
# with open(utils.OUTPUT / "splits"/ "TV_splits.json", "w") as f:
#     json.dump(splits, f)



PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
transform_BERT = data.TransformBERT(tokenizer = tokenizer,
                                     start=0,
                                     fixed_len=400,
                                     min_len=None,
                                     )

ds_train = data.PublisherDataset(publisher="NOZ", set = "train", target = "avgTimeOnPage", text_base = "article_text", transform = transform_BERT)
print(len(ds_train))
ds_dev = data.PublisherDataset(publisher="NOZ", set = "dev", target = "avgTimeOnPage", text_base = "article_text", transform = transform_BERT)
print(len(ds_dev))
ds_test = data.PublisherDataset(publisher="NOZ", set = "test", target = "avgTimeOnPage", text_base = "article_text", transform = transform_BERT)
print(len(ds_test))

BATCH_SIZE = 8
collater_BERT = data.CollaterBERT()

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collater_BERT)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collater_BERT)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=collater_BERT)


# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
print(data['articleId'])
print(data['publisher'])
print(data['target'])
print(data['textlength'])

