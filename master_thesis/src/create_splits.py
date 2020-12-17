from master_thesis.src import models, data, utils
import json

#####

NOZ = utils.get_publisher_df("NOZ")
print(NOZ.head())

splits = data.create_train_dev_test(df = NOZ, random_seed=123)
with open(utils.OUTPUT / "splits" / "NOZ_splits.json", "w") as f:
    json.dump(splits, f)

#####

SZ = utils.get_publisher_df("SZ")
print(SZ.head())

splits = data.create_train_dev_test(df = SZ, random_seed=123)
with open(utils.OUTPUT / "splits"/ "SZ_splits.json", "w") as f:
    json.dump(splits, f)

#####

TV = utils.get_publisher_df("TV")
print(TV.head())

splits = data.create_train_dev_test(df = TV, random_seed=123)
with open(utils.OUTPUT / "splits"/ "TV_splits.json", "w") as f:
    json.dump(splits, f)

#####

bonn = utils.get_publisher_df("bonn")
print(bonn.head())

splits = data.create_train_dev_test(df = bonn, random_seed=123)
with open(utils.OUTPUT / "splits"/ "bonn_splits.json", "w") as f:
    json.dump(splits, f)




