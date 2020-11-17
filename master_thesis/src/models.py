from master_thesis.src import utils
import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased' # 'distilbert-base-german-cased'

#### deprecated, now use Bert_sequence
#def get_model_and_tokenizer(pretrained = PRE_TRAINED_MODEL_NAME): # 'distilbert-base-german-cased'
#    """
#    from BertForSequenceClassification-Documentation:
#    returns:
#        loss: only when `label` is provided
#        logits: shape (batch_size, config.num_labels)
#        hidden_states: only returned when ``output_hidden_states=True``
#        attentions: only returned when ``output_attentions=True``
#    """
#
#    model = BertForSequenceClassification.from_pretrained(pretrained, # use path to load saved model, otherwise PRE_...
#                                                          num_labels = 1, # turns "classification" into regression?
#                                                          output_attentions = False,
#                                                          output_hidden_states = False,
#                                                         )
#    # config.hidden_dropout_prob =  0.1 #TODO: das ist nicht sonderlich viel
#
#    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#
#    return model, tokenizer


class Bert_sequence(nn.Module):
    """Basically BertForSequenceClassification, but it only outputs the logits.
    """

    def __init__(self, n_outputs):
        super(Bert_sequence, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                                  num_labels = n_outputs,
                                                                  output_attentions = False,
                                                                  output_hidden_states = False)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        return out[0] # just the logits


# das hier ist im Wesentlichen (glaube ich...) das gleiche, aber mit anderem Dropout und einem extra Intermediate Linea Layer
class Bert_regression(nn.Module):
    """ This uses BertModel, applies own dropout and adds two linear Layers

    from BertModel_Documentation:
    returns:
        Return:
        last_hidden_state: shape(batch_size, sequence_length, hidden_size)
        pooler_output: (batch_size, hidden_size):
            Last layer hidden-state of the first token of the sequence (classification token) #TODO: Sollte ich das ernst nehmen?
            This output is usually *not* a good summary of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states: only returned when ``output_hidden_states=True``
        attentions: only returned when ``output_attentions=True``
    """

    def __init__(self, n_outputs):
        super(Bert_regression, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3) # ich glaube, bei BertForSequenceClassification ist das 0.1
        #self.fc = nn.Linear(self.bert.config.hidden_size, 128) # (hidden_size = 768) # das hat BertForSequenceClassification nicht
        self.out = nn.Linear(self.bert.config.hidden_size, n_outputs) # 128

    def forward(self, input_ids, attention_mask):
        last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask) [1] # stimmt das? ist trotzdem nur das vom CLS, oder?
        last_hidden = self.drop(last_hidden) # Dropout
        #inter = self.fc(last_hidden)
        #out = self.out(inter)
        out = self.out(last_hidden)
        return out


# Das hier nimmt nicht den last hidden state des CLS-Tokens als Sequence-Repräsentation, sondern mittelt über alle
class Bert_averaging(nn.Module):
    def __init__(self, n_outputs):
        super(Bert_averaging, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_hidden_states=True) # important!

        self.drop = nn.Dropout(p=0.3) # ich glaube, bei BertForSequenceClassification ist das 0.1
        self.out = nn.Linear(self.bert.config.hidden_size, n_outputs) # 128

    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask) [0] # stimmt das?
        #print(hidden_states.size())
        avg_hidden = torch.mean(hidden_states, dim=1) # averaging the last hidden states of ALL tokens in sequence
                                                       # instead of taking just the one of CLS-token
        #print(avg_hidden.size())
        avg_hidden = self.drop(avg_hidden) # Dropout
        out = self.out(avg_hidden)
        #print(out.size())
        return out


# not yet used, maybe something to try?
#class DistilBert_sequence(Bert_sequence):
#    """ uses everything from Bert_sequence, but uses DistilBert instead of Bert
#    ACHTUNG: Der Tokenizer verwendet andere special tokens, aber das Dataset sollte das richtig machen...
#    """
#    def __init__(self, n_outputs):
#        super(Bert_sequence, self).__init__()
#        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-german-cased',
#                                                                        num_labels=n_outputs,
#                                                                        output_attentions=False,
#                                                                        output_hidden_states=False)


# (fast) identisch nachgebaut wie hier https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN(nn.Module):
    def __init__(self, num_outputs,
                       embs_dim = 300,
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

        ## Max pooling. Output shape: (b, num_filters[i], 1)
        #x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #               for x_conv in x_conv_list]

        # Average pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.avg_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Output shape: (b, n_classes)
        out = self.fc(self.drop(x_fc))

        return out


# basically the same as above but smaller and with an intermediate linear layer
class CNN_small(nn.Module):
    def __init__(self, num_outputs,
                       embs_dim = 300,
                       filter_sizes=[3, 4, 5],
                       num_filters=[20,20,20]
                 ):
        super(CNN_small, self).__init__()
        self.embs_dim = embs_dim
        self.num_outputs = num_outputs
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.conv1d_list = nn.ModuleList([ nn.Conv1d(in_channels=self.embs_dim,
                                                     out_channels=self.num_filters[i],
                                                     kernel_size=self.filter_sizes[i])
                                           for i in range(len(self.filter_sizes))
                                         ])

        self.fc = nn.Linear(np.sum(self.num_filters), 32)
        self.out = nn.Linear(32, self.num_outputs)
        self.drop = nn.Dropout(p=0.4) # hier war sonst IMMER 0.5, mal kleiner ausprobiert
        self.drop_embs = nn.Dropout(p=0.2)

    def forward(self, x):
        x_embed = self.drop_embs(x)
        x_reshaped = x_embed.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [F.avg_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        inter = self.fc(self.drop(x_fc))
        out = self.out(self.drop(inter))
        return out



class FFN_BERT(nn.Module):
    """Combines FFN and Bert: Concatenating Bert-Output and meta data.
    """

    def __init__(self, n_outputs):
        super(FFN_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_attentions = False,
                                              output_hidden_states = True,
                                              )

        self.publisher_embs = nn.Embedding(5, 100)

        self.ffn = nn.Sequential(nn.Linear(869, 256), # 869 = 768 (Bert) + 100 (Embs) + 1 (textlength)
                                 nn.LeakyReLU(0.01),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(0.01),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(128, 64),
                                 nn.LeakyReLU(0.01),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(64, n_outputs)
                                 )


    def forward(self, input_ids, attention_mask, textlength, publisher):
        out_bert = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        hidden_state_cls = out_bert[1] # pooled output (hidden state of first token with some modifications)

        publisher = self.publisher_embs(publisher).squeeze()
        concatenated = torch.cat([hidden_state_cls, publisher, textlength], dim=1)

        out = self.ffn(concatenated)
        return out


class baseline(nn.Module):
    """Just the FFN with textlength and publisher.
    """

    def __init__(self, n_outputs):
        super(baseline, self).__init__()

        self.LReLU = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.3)

        self.publisher_embs = nn.Embedding(5, 100)
        self.fc1 = nn.Linear(101, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_outputs)

    def forward(self, textlength, publisher):
        publisher = self.publisher_embs(publisher).squeeze()
        concatenated = torch.cat([publisher, textlength], dim=1)
        out_fc1 = self.dropout(self.LReLU(self.fc1(concatenated)))
        out_fc2 = self.dropout(self.LReLU(self.fc2(out_fc1)))
        out_fc3 = self.dropout(self.LReLU(self.fc3(out_fc2)))
        out = self.out(out_fc3)

        return out

class baseline_in_FFN_BERT(baseline):
    """
    Erbt von baseline (hat daher gleiche Layernamen und Parameter),
    ABER fügt Bert zu einem der Layer in forward-hinzu hinzu.
    """

    def forward(self, textlength, publisher, bert_output):
        publisher = self.publisher_embs(publisher).squeeze()
        concatenated = torch.cat([publisher, textlength], dim=1)
        out_fc1 = self.dropout(self.LReLU(self.fc1(concatenated)))
        out_fc1_bert = out_fc1 + bert_output                          # adds Bert-output (modified with one linear layer)
                                                                      # size 256
        out_fc2 = self.dropout(self.LReLU(self.fc2(out_fc1_bert)))
        out_fc3 = self.dropout(self.LReLU(self.fc3(out_fc2)))
        out = self.out(out_fc3)

        return out

class baseline_in_FFNBERTFeatures(baseline):

    def __init__(self, n_outputs):
        super(baseline_in_FFNBERTFeatures, self).__init__(n_outputs=1)
        self.fc1 = nn.Linear(869, 256)


    def forward(self, textlength, publisher, bert_output):
        publisher = self.publisher_embs(publisher).squeeze()
        concatenated = torch.cat([bert_output, publisher, textlength], dim=1) # bert_output is avg of last hidden states
        #print(concatenated.size())
        out_fc1 = self.dropout(self.LReLU(self.fc1(concatenated))) # from size 869 to 256
        out_fc2 = self.dropout(self.LReLU(self.fc2(out_fc1)))
        out_fc3 = self.dropout(self.LReLU(self.fc3(out_fc2)))
        out = self.out(out_fc3)

        return out


class FFN_BERT_pretrained(nn.Module):
    """Combines FFN and Bert: Using parameters of pretrained baseline, adding BERT output.
    """

    def __init__(self, n_outputs):
        super(FFN_BERT_pretrained, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_attentions = False,
                                              output_hidden_states = True)
        self.fc_bert = nn.Linear(768, 256)
        self.LReLU = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.3)
        self.baseline_in_FFN_BERT = baseline_in_FFN_BERT(n_outputs=1)


    def forward(self, input_ids, attention_mask, textlength, publisher):
        out_bert = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        hidden_state_cls = out_bert[1] # pooled output (last hidden state of first token with some modifications)
        #hidden_state_cls = out_bert[0][:,0,:] # last hidden state of first token (no modifications)
        # auch hier stattdessen mal Mitteln über die last hidden states aller Tokens ausprobieren?

        bert_output = self.dropout(self.LReLU(self.fc_bert(hidden_state_cls)))

        out = self.baseline_in_FFN_BERT(textlength=textlength, publisher=publisher, bert_output=bert_output)

        return out

class FFN_BERTFeatures(nn.Module):

    def __init__(self, n_outputs):
        super(FFN_BERTFeatures, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_attentions = False,
                                              output_hidden_states = True)

        self.baseline_in_FFNBERTFeatures = baseline_in_FFNBERTFeatures(n_outputs=1)


    def forward(self, input_ids, attention_mask, textlength, publisher):
        out_bert = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        last_hidden_states = out_bert[0]  # stimmt das?
        #print(last_hidden_states.size())
        avg_last_hidden = torch.mean(last_hidden_states, dim=1)  # averaging the last hidden states of ALL tokens in sequence
        #print(avg_last_hidden.size())

        out = self.baseline_in_FFNBERTFeatures(textlength = textlength, publisher=publisher, bert_output = avg_last_hidden)
        return out


# mal anders: die Metadaten in Bert unten reinschieben, also auf Bert-Dimension bringen (FFN) und dann als sozusagen zwei "zusätzliche" Wörter für
# Bert behandeln, also irgendwie auf die unterste Bert-Layer als "Wort" bringen.
# publisher_EMbedings sind das schon
# Textlänge mit FFN
# dazu müsste man wohl so richtig in Bert rumschnippeln, also in den huggingface-Methoden an die Embedding-Methode Dinge dranhängen (erben etc)

# timestep-wise dropout probieren? gemeint: Wörter im Text maskieren (Iyyer deep composition rivals...
# https://www.aclweb.org/anthology/P15-1162/
# word dropout
# attention mask ändern! dropout auf attention-mask ist möglicherweise schon alles!

# das dann natürlich auch (zuerst) auch in einer Bert-Baseline machen, also Wort-dropout, scheduler, learning-rate ändern
# aaaaaaaaaah

# aus der Signal-Nachricht (10.November): Bert einfach als Feature-Extraktor nehmen und nicht updaten!