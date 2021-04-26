from master_thesis.src import utils
import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased' # 'distilbert-base-german-cased'


def get_BertModel():
    bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                     output_attentions=False,
                                     output_hidden_states=True)
    return bert


def get_BertForSequenceClassification(n_outputs):
    bertSequence = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                                 num_labels = n_outputs,
                                                                 output_attentions = False,
                                                                 output_hidden_states = False)
    return bertSequence


def get_FFN(n_outputs, nr_layers = 2, input_size = 768, p_drop = 0.3):

    if nr_layers == 0:
        ffn = nn.Sequential(nn.Linear(input_size, n_outputs))

    if nr_layers == 1:
        ffn = nn.Sequential(nn.Linear(input_size, 256),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(256, n_outputs),
                            )

    if nr_layers == 2:
        ffn = nn.Sequential(nn.Linear(input_size, 256),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(256, 64),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(64, n_outputs)
                            )

    if nr_layers == 3:
        ffn = nn.Sequential(nn.Linear(input_size, 256),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(256, 128),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(128, 64),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(64, n_outputs)
                            )

    if nr_layers == 4:
        ffn = nn.Sequential(nn.Linear(input_size, 512),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(256, 128),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(128, 64),
                            nn.LeakyReLU(0.01),
                            nn.Dropout(p=p_drop),
                            nn.Linear(64, n_outputs)
                            )

    return ffn


class EmbsFFN(nn.Module):
    """
    input: document embedding vector (averageds word embs or BERT) --> FFN
    input_size: size of input vector (fastText: 300, BERT: 768)
    """
    def __init__(self, input_size, n_outputs):
        super(EmbsFFN, self).__init__()
        self.ffn = get_FFN(n_outputs=n_outputs, input_size=input_size, nr_layers=2)

    def forward(self, vector):
        out = self.ffn(vector)
        return out


##### multiple ways of Bert baseline (no textlength) #####

class BertSequence(nn.Module):
    """ simply BertForSequence (no non-linearity)
    """
    def __init__(self, n_outputs):
        super(BertSequence, self).__init__()
        self.bertSequence = get_BertForSequenceClassification(n_outputs=1)

    def forward(self, input_ids, attention_mask):
        out_bert = self.bertSequence(input_ids=input_ids, attention_mask=attention_mask)
        return out_bert[0] # logits...


class BertFFN(nn.Module):
    """ BertModel, own FFN on top of hidden state of cls token
    """
    def __init__(self, n_outputs):
        super(BertFFN, self).__init__()
        self.bert = get_BertModel()
        #self.ffn = get_FFN(n_outputs=n_outputs, input_size=768, nr_layers=1)
        self.ffn = get_FFN(n_outputs=n_outputs, input_size=768, nr_layers=2)
        #self.ffn = get_FFN(n_outputs=n_outputs, input_size=768, nr_layers=3)

    def forward(self, input_ids, attention_mask):
        out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state_cls = out_bert[1]
        out = self.ffn(hidden_state_cls)
        return out


class BertAveraging(nn.Module):
    """ does use ALL last hidden states (not just from CLS) and takes average,
        then linear layers
    """
    def __init__(self, n_outputs):
        super(BertAveraging, self).__init__()
        self.bert = get_BertModel()
        self.ffn = get_FFN(n_outputs=n_outputs, input_size=768, nr_layers=2)

    def forward(self, input_ids, attention_mask):
        out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out_bert[0] # stimmt das?
        avg_hidden = torch.mean(hidden_states, dim=1) # averaging the last hidden states of ALL tokens in sequence
                                                       # instead of taking just the one of CLS-token
        out = self.ffn(avg_hidden)
        return out





class BertTextlength(nn.Module):
    """
    here: BertModel
    BERT output concatenated with textlength
    then FFN
    """

    def __init__(self, n_outputs):
        super(BertTextlength, self).__init__()

        self.bert = get_BertModel()
        #self.ffn = get_FFN(n_outputs=n_outputs, input_size=769, nr_layers=1)
        self.ffn = get_FFN(n_outputs=n_outputs, input_size=769, nr_layers=2) # TODO: das hier war bisher
        #self.ffn = get_FFN(n_outputs=n_outputs, input_size=769, nr_layers=3)
        #self.ffn = get_FFN(n_outputs=n_outputs, input_size=769, nr_layers=4)

    def forward(self, input_ids, attention_mask, textlength):
        out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state_cls = out_bert[1]  # pooled output (hidden state of first token with some modifications)
        concatenated = torch.cat([hidden_state_cls, textlength], dim=1)
        out = self.ffn(concatenated)

        return out




class baseline_textlength(nn.Module):
    """ no BERT!
        an FFN with JUST textlength
    """
    def __init__(self, n_outputs):
        super(baseline_textlength, self).__init__()

        # self.ffn = nn.Sequential(nn.Linear(1, 32),
        #                          nn.LeakyReLU(0.01),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(32, 64),
        #                          nn.LeakyReLU(0.01),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(64, 32),
        #                          nn.LeakyReLU(0.01),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(32, n_outputs)
        #                          )

        self.ffn = nn.Sequential(nn.Linear(1, 32),
                                 nn.ReLU(), #        nn.LeakyReLU(0.01),
                                 nn.Linear(32, n_outputs)
                                 )

        # self.ffn = nn.Sequential(nn.Linear(1, n_outputs))

    def forward(self, textlength):
        out = self.ffn(textlength)
        return out



class BertHierarchical(nn.Module):
    """
    split text in chunks of <=512 tokens and combine outputs somehow
    todo: should textlength be used or not? --> gerade ist es nicht drin
    """

    def __init__(self, n_outputs, max_sect = 5):
        super(BertHierarchical, self).__init__()
        self.max_sect = max_sect

        self.bert = get_BertModel()
        self.ffn = get_FFN(n_outputs = n_outputs, nr_layers= 2, input_size= 768)
        self.weight_vector = torch.nn.Parameter(torch.ones(self.max_sect), requires_grad=True)
                        # learnable weight vector for weighted sum/mean of section outputs

    def forward(self, section_input_ids, section_attention_mask):
        # Sven meint: hier so tun als wäre input shape (max_sect, Bert-Ids) also  batch ignorieren
        batch_size = section_input_ids.size()[0]
        nr_sections = section_input_ids.size()[1]
        sections_out_dummy = torch.zeros(batch_size, self.max_sect, 1,
                                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # to be filled (batch_size, max_sect, 1)
        #sections_out = []

        ### Sven meint so: (das klappt aber nicht, die "zwei zu ignorierenden Dimensionen" scheinen zu stören
        #out_bert = self.bert(input_ids = section_input_ids, attention_mask = section_attention_mask)
        #section_out = out_bert[0]
        #print("Sven section out", section_out)

        for nr in range(nr_sections):
            input_ids = section_input_ids[:,nr,:]
                #input_ids = section_input_ids[nr,:]

            attention_mask = section_attention_mask[:,nr,:]
                #attention_mask = section_attention_mask[nr,:]

            out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            ### wenn BertModel:
            hidden_state_cls = out_bert[1]  # pooled output wenn BertModel (hidden state of first token with some modifications)
            section_out = self.ffn(hidden_state_cls)

            sections_out_dummy[:,nr,:] = section_out
                #sections_out_dummy[nr,:] = section_out
                #sections_out.append(section_out)

            #sections_out = torch.stack(sections_out, dim=1)
        sections_out = sections_out_dummy
        #print(sections_out)
        #print(self.weight_vector)

        # SUM
        ###section_sum = torch.sum(sections_out, dim=1) # this would be normal sum WITHOUT weighting
        #section_sum = torch.sum(sections_out.squeeze(2)*self.weight_vector, dim=1).unsqueeze(1)
        #return section_sum

        # MEAN
        #section_mean = torch.mean(sections_out.squeeze(2) * self.weight_vector, dim=1).unsqueeze(1)
        section_mean = torch.sum(sections_out.squeeze(2) * self.weight_vector, dim=1).unsqueeze(1)/torch.sum(self.weight_vector)
        #print(section_mean)
        return section_mean


class BERT_embedding(nn.Module):
    """
    just returns document embeddings (hidden state of cls-Token)
    """

    def __init__(self):
        super(BERT_embedding, self).__init__()
        self.bert = get_BertModel()

    def forward(self, input_ids, attention_mask):
        out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state_cls = out_bert[1]

        return hidden_state_cls



class BertHierarchicalRNN(nn.Module):
    """
    split text in chunks of <=512 tokens and combine outputs somehow
    """

    def __init__(self, n_outputs, max_sect = 5):
        super(BertHierarchicalRNN, self).__init__()
        self.max_sect = max_sect
        self.bert = get_BertModel()

        # rnn Layer (input_size, hidden_size, nr_layers)
        #self.rnn = nn.RNN(768, 512, 1, batch_first=True) # TODO: wieviele layers? welche hidden size?
        self.rnn = nn.GRU(768, 768, 2, batch_first=True)
        #self.rnn = nn.LSTM(768, 512, 2, batch_first=True)

        self.ffn = get_FFN(n_outputs=1, input_size=768, nr_layers=2)

    def forward(self, section_input_ids, section_attention_mask):
        nr_sections = section_input_ids.size()[1]
        sections_out = []
        #print("nr_sections in this batch:", nr_sections)
        for nr in range(nr_sections):
            #print("nr", nr)
            input_ids = section_input_ids[:,nr,:]
            attention_mask = section_attention_mask[:,nr,:]
            out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state_cls = out_bert[1]  # pooled output (hidden state of first token with some modifications)
            #print(hidden_state_cls.size())
            sections_out.append(hidden_state_cls)
        sections_out = torch.stack(sections_out, dim=1) # (batchsize, sections, 768)
        #print(sections_out)
        #print("sections_out", sections_out.size())

        # now an rnn over all section_out per section
        output, hidden_state_last_timestep = self.rnn(sections_out)
        #print("output", output.size())

        # pytorch gives hidden_state_last_timestep of ALL layers --> take the one from the last layer ([-1])
        #print("hidden_state_last_timestep", hidden_state_last_timestep.size())
        #todo: ist das so richtig?

        out = self.ffn(hidden_state_last_timestep[-1,:,:]) # last hidden state, last timestep, LAST LAYER
        #print("out", out.size())

        return out







######
###### CNN Versuche
######


# (fast) identisch nachgebaut wie hier https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN(nn.Module):
    """ uses dataloader that returns input_matrix von input_ids of the text
    """
    def __init__(self, n_outputs,
                       embs_dim = 300,
                       filter_sizes=[2, 3, 4, 5],
                       num_filters=64):
        super(CNN, self).__init__()
        self.embs_dim = embs_dim
        self.n_outputs = n_outputs
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # convolutional filters of different kernelsizes
        # using conv2d
        #self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,
        #                                      out_channels = self.num_filters,
        #                                      kernel_size = (filter_size, self.embs_dim),
        #                                      )
        #                                for filter_size in self.filter_sizes])

        # using conv1d
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.embs_dim,
                                              out_channels=self.num_filters,
                                              kernel_size=filter_size)
                                          for filter_size in self.filter_sizes])


        self.drop = nn.Dropout(p=0.2)
        self.LReLU = nn.LeakyReLU(0.01)

        self.ffn = nn.Sequential(nn.Linear(self.num_filters * len(self.filter_sizes), 64),
                                 self.LReLU,
                                 self.drop,
                                 nn.Linear(64, self.n_outputs)
                                 )

        #self.ffn = nn.Sequential(nn.Linear(self.num_filters * len(self.filter_sizes), 1),
        #                         )

    def forward(self, input_matrix):
        x = self.drop(input_matrix)

        # change shape to match shape requirement...
        #x = torch.unsqueeze(x, dim=1) # for 2d
        x = x.permute(0, 2, 1) # for 1d
        #print("x after reshaping", x.size())

        # Apply CNN and activation. Output shape: (b, num_filters, L_out)
        x_conv_list = [self.LReLU(conv(x)) for conv in self.convs]
        #print("x_conv_list", x_conv_list[0].size(), x_conv_list[1].size(), x_conv_list[2].size(), x_conv_list[3].size())


        ## Max pooling. Output shape: (b, num_filters, 1)
        x_pool_list = [F.max_pool1d(torch.squeeze(x_conv, -1), kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Average pooling. Output shape: (b, num_filters, 1)
        #x_pool_list = [F.avg_pool1d(torch.squeeze(x_conv, -1), kernel_size=x_conv.shape[2])
        #               for x_conv in x_conv_list]

        #print("x_pool_list", x_pool_list[0].size(), x_pool_list[1].size(), x_pool_list[2].size(), x_pool_list[3].size())


        # Concatenate x_pool_list to feed into FFN
        x_flat = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        #print("x_flat", x_flat.size())
        out = self.ffn(x_flat)

        return out


# (fast) identisch nachgebaut wie hier https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN_textlength(nn.Module):
    """ same as the CNN but adds textlength info
    """
    def __init__(self, num_outputs,
                       embs_dim = 300,
                       filter_sizes=[3, 4, 5],
                       num_filters=[100,100,100]
                 ):
        super(CNN_textlength, self).__init__()
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
        self.fc = nn.Linear(np.sum(self.num_filters), 128)
        self.out = nn.Linear(129, self.num_outputs) # before this: concatenate textlength
        self.drop = nn.Dropout(p=0.5)
        self.drop_embs = nn.Dropout(p=0.2)
        self.LReLU = nn.LeakyReLU(0.01)


    def forward(self, input_matrix, textlength):
        x_embed = self.drop_embs(input_matrix)
        x_reshaped = x_embed.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [F.avg_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        x_fc2 = self.drop(self.LReLU(self.fc(x_fc)))
        concatenated = torch.cat([x_fc2, textlength], dim=1)
        out = self.out(concatenated)

        return out




# class BERT_textlength(nn.Module):
#     """
#     here: BertModel
#     pooled BERT output concatenated with textlength
#     """
#
#     def __init__(self, n_outputs):
#         super(BERT_textlength, self).__init__()
#         self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
#                                               output_attentions = False,
#                                               output_hidden_states = True)
#
#         self.ffn = nn.Sequential(nn.Linear(769, 256),  #768 (Bert) + 1 (textlength)
#                                  nn.LeakyReLU(0.01),
#                                  nn.Dropout(p=0.3),
#                                  nn.Linear(256, 128),
#                                  nn.LeakyReLU(0.01),
#                                  nn.Dropout(p=0.3),
#                                  nn.Linear(128, 64),
#                                  nn.LeakyReLU(0.01),
#                                  nn.Dropout(p=0.3),
#                                  nn.Linear(64, n_outputs)
#                                  )
#
#     def forward(self, input_ids, attention_mask, textlength):
#         out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_state_cls = out_bert[1]  # pooled output (hidden state of first token with some modifications)
#
#         concatenated = torch.cat([hidden_state_cls, textlength], dim=1)
#         out = self.ffn(concatenated)
#
#         return out




######
###### das hier sind alte Modelle, die noch Publisher als Feature benutzen, aktuell betrachte ich ja alle Publisher einzeln ######
######


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

        self.ffn = nn.Sequential(nn.Linear(869, 256), # 869 = 768 (Bert) + 100 (publisher Embs) + 1 (textlength)
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
    """ an FFN with textlength and publisher
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
        concatenated = torch.cat([bert_output, publisher, textlength], dim=1) # bert_output is avg of last hidden states (size 768)
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







# class MyBertModel(BertModel):
#
#     @classmethod
#     def from_pretrained(cls):
#         return super().from_pretrained(PRE_TRAINED_MODEL_NAME,
#                                               output_attentions=False,
#                                               output_hidden_states=True)

# class LayerBert(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
#                                               output_attentions=False,
#                                               output_hidden_states=True)
#     def forward(self, x):
#         return self.bert(x)


# class LayerBertSequence(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         self.bertSequence = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,
#                                                                   num_labels = n_outputs,
#                                                                   output_attentions = False,
#                                                                   output_hidden_states = False)
#     def forward(self, x):
#         return self.bertSequence(x)






# class LayerBertOut(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         # self.bert_ffn = nn.Sequential(nn.Linear(768, 256),  # 768 (Bert hidden state size)
#         #                      nn.LeakyReLU(0.01),
#         #                      nn.Dropout(p=0.3),
#         #                      nn.Linear(256, 128),
#         #                      nn.LeakyReLU(0.01),
#         #                      nn.Dropout(p=0.3),
#         #                      nn.Linear(128, 64),
#         #                      nn.LeakyReLU(0.01),
#         #                      nn.Dropout(p=0.3),
#         #                      nn.Linear(64, n_outputs)
#         #                      )
#         self.bert_ffn = nn.Sequential(nn.Linear(768, 256),  # 768 (Bert hidden state size)
#                              nn.LeakyReLU(0.01),
#                              nn.Dropout(p=0.3),
#                              nn.Linear(256, 64),
#                              nn.LeakyReLU(0.01),
#                              nn.Dropout(p=0.3),
#                              nn.Linear(64, n_outputs)
#                              )
#
#     def forward(self, x):
#         return self.bert_ffn(x)


# class LayerFFN(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         self.ffn = nn.Sequential(nn.Linear(2, 10),
#                              nn.LeakyReLU(0.01),
#                              nn.Dropout(p=0.3),
#                              nn.Linear(10, 5),
#                              nn.LeakyReLU(0.01),
#                              nn.Dropout(p=0.3),
#                              nn.Linear(5, n_outputs)
#                              )
#     def forward(self, x):
#         return self.ffn(x)


# class LayerBertTextlength(nn.Module):
#     def __init__(self, n_outputs):
#         super().__init__()
#         self.ffn = nn.Sequential(nn.Linear(769, 10),
#                              nn.LeakyReLU(0.01),
#                              nn.Dropout(p=0.3),
#                              nn.Linear(10, 5),
#                              nn.LeakyReLU(0.01),
#                              nn.Dropout(p=0.3),
#                              nn.Linear(5, n_outputs)
#                              )
#     def forward(self, x):
#         return self.ffn(x)