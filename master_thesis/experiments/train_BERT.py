import torch
from torch import optim, nn
import pandas as pd
import numpy as np

from master_thesis.src import utils, data, models

assert torch.cuda.is_available()
device = torch.device('cuda:0')
print("Device is: ", device)

# get pretrained model and tokenizer
model, tokenizer = models.get_model_and_tokenizer()
model.to(device)

# try out tokenizer
sample_text = "Das hier ist ein deutscher Beispieltext. Und einen zweiten müssen wir auch noch haben."
#tokens = tokenizer.tokenize(sample_text) # just tokenizes
#token_ids = tokenizer.convert_tokens_to_ids(tokens)
#ids = tokenizer.encode(sample_text) # already adds special tokens
encoded_plus = tokenizer.encode_plus(sample_text,
                                     max_length = 10,
                                     return_token_type_ids=False,
                                     pad_to_max_length=True,
                                     truncation=True,
                                     return_attention_mask=True,)

#print(tokens)
#print(token_ids)
#print(ids)
print("Testing the tokenizer:" , encoded_plus)

#tokenizer.get_vocab() # shows tokenizer vocab (subwords!)
#tokenizer.sep_token, tokenizer.sep_token_id, tokenizer.cls_token, tokenizer.cls_token_id, tokenizer.pad_token, tokenizer.pad_token_id


# get raw data
df = pd.read_csv(utils.DATA / 'combined.tsv', sep = '\t')
df = df.fillna('') # replacing Nan with emtpy string
print("Shape of raw df:", df.shape)

# just take articles with ...
df = df.loc[(df['pageviews'] >= 20) &
            (df['avgTimeOnPagePerNr_tokens'] <= 4) &
            (df['avgTimeOnPagePerNr_tokens'] >= 0.01)
            ]
print("Remaining df after conditioning:", df.shape)

# building train-dev-test split, their DataSets and DataLoaders

BATCH_SIZE = 12
dl_train, dl_dev, dl_test = data.create_DataLoaders(df = df,
                                                    target = 'avgTimeOnPagePerNr_tokens',
                                                    text_base = 'text_preprocessed', # 'titelH1',
                                                    tokenizer = tokenizer,
                                                    max_len = 100,            # change depending on used text_base!
                                                    batch_size = BATCH_SIZE)

# have a look at one batch in dl_train to see if shapes make sense
data = next(iter(dl_train))
print(data.keys())
input_ids = data['input_ids']
#print(input_ids)
print(input_ids.shape)
attention_mask = data['attention_mask']
#print(attention_mask)
print(attention_mask.shape)
print(data['target'].shape)

# loss and optimizer

optimizer = optim.Adam(model.parameters(), lr=1e-5)
#optimizer = optim.AdamW(net.parameters(), lr=0.001)
#optimizer = AdamW(model.parameters(),lr=1e-5)

loss_fn = nn.MSELoss()  # mean squared error

##### TRAINING AND EVALUATING #####

EPOCHS = 10

for epoch in range(EPOCHS):
    print("Epoch", epoch)

    ### TRAINING on train
    print("training")
    model = model.train()
    train_losses = []

    for nr, d in enumerate(dl_train):
        print("-Batch", nr, end='\r')
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)
        # print(targets.shape)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]  # stimmt das so? ist [0] die logits?
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # print(np.mean(train_losses))
    print("Mean train loss:", np.mean(train_losses))

    ### EVALUATING on dev
    print("evaluating")
    model = model.eval()
    eval_losses = []

    with torch.no_grad():
        for nr, d in enumerate(dl_dev):
            print("-Batch", nr, end='\r')
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]  # stimmt das so?

            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())
            # print(np.mean(eval_losses))
        print("Mean eval loss:", np.mean(eval_losses))
