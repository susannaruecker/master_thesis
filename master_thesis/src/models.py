from master_thesis.src import utils
import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

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


# https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
# scheint mir ein gutes Tutorial, vielleicht so nachbauen?
class CNN(nn.Module):
    def __init__(self, num_outputs,
                       embs_dim,
                       filter_sizes=[3, 4, 5],
                       num_filters=[100,100,100]
                       ):
        super(CNN, self).__init__()
        self.embs_dim = embs_dim
        self.num_outputs = num_outputs
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # Convolutional Filters
        self.conv1d_list = nn.ModuleList([ nn.Conv1d(in_channels=self.embs_dim,
                                                     out_channels=self.num_filters[i],
                                                     kernel_size=self.filter_sizes[i])
                                           for i in range(len(self.filter_sizes))
                                         ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(self.num_filters), self.num_outputs)
        self.drop = nn.Dropout(p=0.5)
        self.drop_embs = nn.Dropout(p=0.2)



    def forward(self, x):

        # x is already embedding matrix. Output shape: (b, max_len, embed_dim)
        x_embed = self.drop_embs(x)

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Output shape: (b, n_classes)
        out = self.fc(self.drop(x_fc))

        return out


