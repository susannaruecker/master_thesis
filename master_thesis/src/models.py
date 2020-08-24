from master_thesis.src import utils
import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from torch import nn, optim

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased' # 'distilbert-base-german-cased'

def get_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                          num_labels = 1, # turns "classification" into regression?
                                                          output_attentions = False,
                                                          output_hidden_states = False,
                                                         )
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    return model, tokenizer



# das ist umständlich (und auch falsch), ich habe stattdessen bereits BertForSequenceClassification genommen
# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

#class Bert_regression(nn.Module):
#
#    def __init__(self, n_outputs): # maybe train pageviews and timeOnPage simultaneously?
#        super(Bert_regression, self).__init__()
#        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
#        self.drop = nn.Dropout(p=0.3)
#        self.out = nn.Linear(self.bert.config.hidden_size, n_outputs)
#
#    def forward(self, input_ids, attention_mask):
#        _, pooled_output = self.bert(input_ids=input_ids,           # das hier ist glaube ich nicht sinnvoll bei mir
#                                     attention_mask=attention_mask)
#        output = self.drop(pooled_output)
#        return self.out(output)

#%%

#model = Bert_regression(n_outputs = 1)
#model = model.to(device)
