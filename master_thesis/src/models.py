from master_thesis.src import utils
import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from torch import nn, optim
import torch.nn.functional as F

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased' # 'distilbert-base-german-cased'

def get_model_and_tokenizer(pretrained = PRE_TRAINED_MODEL_NAME): # 'distilbert-base-german-cased'

    model = BertForSequenceClassification.from_pretrained(pretrained, # use path to load saved model, otherwise PRE_...
                                                          num_labels = 1, # turns "classification" into regression?
                                                          output_attentions = False,
                                                          output_hidden_states = False,
                                                         )
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    return model, tokenizer



# das nachfolgende ist umständlich (und vielleicht auch falsch), ich habe stattdessen bereits BertForSequenceClassification genommen
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



class CNN(nn.Module):
    def __init__(self, num_outputs, fixed_length):  # fixed_length to pad or trim text to
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(300, 128, kernel_size= (1, 300), stride=1, padding=0)
        self.conv2 = nn.Conv2d(300, 128, kernel_size= (2, 300), stride=1, padding=0)
        self.conv3 = nn.Conv2d(300, 128, kernel_size= (3, 300), stride=1, padding=0)
        self.drop_embs = nn.Dropout(p=0.2)  # 0.2 dropout
        self.drop_hidden = nn.Dropout(p=0.5)  # 0.5 dropout
        self.label = nn.Linear(3 * 128, num_outputs)

    def forward(self, x):
        print(x.size())
        # input layer
        x = self.drop_embs(x)
        print(x.size())

        # convolutional layer
        out1 = self.conv1(x)  # (batch_size, out_channels, dim, 1)
        print(out1.size())
        out1 = F.relu(out1.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        print(out1.size())
        out1 = F.max_pool1d(out1, out1.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        print(out1.size())

        #out2 = self.conv2(x)  # (batch_size, out_channels, dim, 1)
        #out2 = F.relu(out2.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        #out2 = F.max_pool1d(out2, out2.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)


        #x = x.unsqueeze(1) # neu
        #print(x.size())
        #x = self.conv1(x)
        #x = F.relu(x)
        #x = self.max_pool(x)  # max pool
        #x = self.drop_hidden(x)  # ist das richtig?

        # dense layer
        #x = x.reshape(-1, x.shape[1] * x.shape[2])  # flatten, but batch_size (= x.shape[0]) should stay; stimmt das so?
        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.drop_hidden(x)

        # output layer
        #x = self.fc2(x)

        #return x
