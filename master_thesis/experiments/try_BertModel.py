from transformers import BertTokenizer, DistilBertTokenizer, BertModel, BertForSequenceClassification
from master_thesis.src import models
import torch

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

text = ["Das hier ist ein Satz.", "Und noch einer", "Und dann sogar noch ein dritter längerer."]

"""
model_regr = models.Bert_regression(n_outputs = 1)
model_Seq = models.Bert_sequence(n_outputs=1)

text = ["Das hier ist ein Satz, den Bert jetzt verstehen soll.", "Und noch einer", "Und dann sogar noch ein dritter, der wieder etwas länger ist."]

inputs = tokenizer(text, padding=True, return_tensors="pt", return_token_type_ids=False)
print(inputs)

print("model_regr")
output1 = model_regr(**inputs)
print(output1.size())
print(output1)

print("model_Seq")
output2 = model_Seq(**inputs)
print(output2.size())
print(output2)

print("DistilBert")
##TODO: Achtung, ich glaube, der DistilBertTokenizer hat andere special tokens
#       das Dataset schaut aber glücklicherweise in die .config des Tokenizers, also wohl kein Problem!
#       beim Collater muss das richtige padding-Symbol verwendet werden, das ist aber wohl überall 0
model_distil = models.DistilBert_sequence(n_outputs=1)
tokenizer_distil = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
inputs_d = tokenizer_distil(text, padding = True, return_tensors='pt', return_token_type_ids=False)
print(inputs_d)
output3 = model_distil(**inputs_d)
print(output3.size())
print(output3)

"""

model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                  output_hidden_states=True)

encoded = tokenizer(text, padding=True, return_tensors="pt", return_token_type_ids=False)
print(encoded)

import numpy as np
outputs = model(**encoded)
print(len(outputs))
print("--------")
print("last hidden states", outputs[0].shape)
avg = torch.mean(outputs[0], dim = 1) # take the mean of the last hidden states of all tokens (instead of CLS-token)
avg2 = torch.mean(outputs[0], axis=1) # lustig: eigentlich heißt es "dim", aber "axis" funktioniert auch! :)
print("avg hidden state", avg.shape)
print(avg)
print("avg hidden state", avg2.shape)
print(avg2)
print(avg)
print("--------")
print("pooler_output", outputs[1].shape)
print(outputs[1])
print("--------")
print("hidden_states embs", outputs[2][0].shape)
#print(outputs[2][0][:,0,:])
print("hidden_states layers", outputs[2][1].shape)
print(outputs[2][1][:,0,:])



#model = models.Bert_averaging(n_outputs=1)
#encoded = tokenizer(text, padding=True, return_tensors="pt", return_token_type_ids=False)
#out = model(**encoded)
#print(out)

#TODO: this seems to be working! now try it out! :D