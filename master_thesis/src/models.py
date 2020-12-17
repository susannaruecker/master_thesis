from master_thesis.src import utils
import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased' # 'distilbert-base-german-cased'


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
    """ uses original BertModel, own dropout and linear layer
    """
    def __init__(self, n_outputs):
        super(Bert_regression, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3) # ich glaube, bei BertForSequenceClassification ist das 0.1
        self.out = nn.Linear(self.bert.config.hidden_size, n_outputs) # 128

    def forward(self, input_ids, attention_mask):
        last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask) [1] # last hidden state from CLS, right?
        last_hidden = self.drop(last_hidden) # Dropout
        out = self.out(last_hidden)
        return out


class Bert_averaging(nn.Module):
    """ does use ALL last hidden states (not just from CLS) and takes average, then dropout and linear layer
    """
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


class Bert_baseline(nn.Module):
    """ just BertModel with an FFN on top (no textlength or publisher info)
    """
    def __init__(self, n_outputs):
        super(Bert_baseline, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_attentions=False,
                                              output_hidden_states=True)
        self.ffn = nn.Sequential(nn.Linear(768, 256),  # 768 (Bert)
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

    def forward(self, input_ids, attention_mask):
        out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state_cls = out_bert[1]  # pooled output (hidden state of CLS token with some modifications)
        out = self.ffn(hidden_state_cls)
        return out


# (fast) identisch nachgebaut wie hier https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN(nn.Module):
    """ uses dataloader that returns input_matrix von input_ids of the text
    """
    def __init__(self, num_outputs,
                       embs_dim = 300,
                       filter_sizes=[3, 4, 5],
                       num_filters=[100,100,100]):
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
        self.fc = nn.Linear(np.sum(self.num_filters), 128)
        self.out = nn.Linear(128, self.num_outputs)
        self.drop = nn.Dropout(p=0.5)
        self.drop_embs = nn.Dropout(p=0.2)
        self.LReLU = nn.LeakyReLU(0.01)

    def forward(self, input_matrix):
        # x is already embedding matrix. Output shape: (b, max_len, embed_dim)
        x_embed = self.drop_embs(input_matrix)
        #print("x_embed", x_embed.size())

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        #print("x_reshaped", x_reshaped.size())

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        #print("x_conv_list_0", x_conv_list[0].size())
        #print("x_conv_list_1", x_conv_list[1].size())
        #print("x_conv_list_2", x_conv_list[2].size())

        ## Max pooling. Output shape: (b, num_filters[i], 1)
        #x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #               for x_conv in x_conv_list]

        # Average pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.avg_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        #print("x_fc", x_fc.size())

        # Output shape: (b, n_classes)
        x_fc2 = self.drop(self.LReLU(self.fc(x_fc)))
        #print("x_fc2", x_fc2.size())
        out = self.out(x_fc2)
        #print("out", out.size())

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


class baseline_textlength(nn.Module):
    """an FFN with JUST textlength
    """

    def __init__(self, n_outputs):
        super(baseline_textlength, self).__init__()

        self.LReLU = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, n_outputs)

    def forward(self, textlength):
        out_fc1 = self.dropout(self.LReLU(self.fc1(textlength)))
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

class BERT_textlength(nn.Module):
    """
    pooled BERT output concatenated with textlength
    """

    def __init__(self, n_outputs):
        super(BERT_textlength, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_attentions = False,
                                              output_hidden_states = True)

        self.ffn = nn.Sequential(nn.Linear(769, 256),  #768 (Bert) + 1 (textlength)
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

    def forward(self, input_ids, attention_mask, textlength):
        out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state_cls = out_bert[1]  # pooled output (hidden state of first token with some modifications)

        concatenated = torch.cat([hidden_state_cls, textlength], dim=1)
        out = self.ffn(concatenated)

        return out


class BERT_hierarchical(nn.Module):
    """
    split text in chunks of <=512 tokens and combine outputs somehow
    """

    def __init__(self, n_outputs, max_sect = 5):
        super(BERT_hierarchical, self).__init__()
        self.max_sect = max_sect
        self.bert = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                                  num_labels=n_outputs,
                                                                  output_attentions=False,
                                                                  output_hidden_states=True) # je nachdem, was man will

        # self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
        #                                       output_attentions = False,
        #                                       output_hidden_states = True)

        self.weight_vector = torch.nn.Parameter(torch.rand(1, self.max_sect), requires_grad=True,
                                                ) # learnable weight vector for weighted sum of section outputs

        ### Das hier ist nötig, wenn man selbstständig mit dem last hidden state vom CLS-Token weitermacht
        ### (Bert-Model statt BertForSequencrClassification)
        # self.ffn = nn.Sequential(nn.Linear(768, 256),  #768 (Bert)
        #                          nn.LeakyReLU(0.01),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(256, 128),
        #                          nn.LeakyReLU(0.01),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(128, 64),
        #                          nn.LeakyReLU(0.01),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(64, n_outputs)
        #                          )

        self.out = nn.Sequential(nn.Linear(2, 10),
                                 nn.LeakyReLU(0.01),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(10, 5),
                                 nn.LeakyReLU(0.01),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(5, n_outputs)
                                 )

    def forward(self, section_input_ids, section_attention_mask, textlength):
        batch_size = section_input_ids.size()[0]
        nr_sections = section_input_ids.size()[1]
        sections_out_dummy = torch.zeros(batch_size, self.max_sect, 1,
                                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # to be filled (batch_size, max_sect, 1)

        #print("nr_sections in this batch:", nr_sections)
        #print(section_input_ids.size())
        #print(section_attention_mask.size())
        for nr in range(nr_sections):
            #print("nr", nr)
            input_ids = section_input_ids[:,nr,:]
            #print(input_ids.size())
            attention_mask = section_attention_mask[:,nr,:]
            #print(attention_mask.size())
            out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            ####hidden_state_cls = out_bert[1]  # if bert = BertModel: pooled output (hidden state of first token with some modifications)
            ####section_out = self.ffn(hidden_state_cls)

            section_out = out_bert[0] # das sind die logits wen bert = BertForSequenceClassification (out_bert[1] wäre last hidden state of CLS)

            #print("section_out", section_out.size())
            #print(section_out)
            sections_out_dummy[:,nr,:] = section_out
        sections_out = sections_out_dummy
        #print("sections_out", sections_out.size())
        #print("sections_out", sections_out)

        #print("weight_vector", self.weight_vector.size())
        #print("weight_vector", self.weight_vector)


        ###section_sum = torch.sum(sections_out, dim=1) # this would be normal sum WITHOUT weighting

        ##### TODO: Das hier ist sehr umständlich, wie geht das besser? Immer dot-product mit weigth_vector aber halt jede batch
        weight_matrix = self.weight_vector.repeat(1,batch_size).view(batch_size, self.max_sect).unsqueeze(1)
        #print("weight_matrix", weight_matrix.size())
        #print(weight_matrix)
        section_sum = torch.bmm(weight_matrix, sections_out).squeeze(1)

        #print("weighted_sum", section_sum.size())
        #print(section_sum)

        #concatenated = torch.cat([section_sum, textlength], dim=1)
        #print("concatenated", concatenated)
        #print(concatenated.size()) # just the "textlength" sum and the textlength

        #out = self.out(concatenated)
        #print("out", out.size())
        #print(out)

        #return out
        return section_sum # TODO: erstmal ohne Textlänge einzubringen

class BERT_hierarchical_RNN(nn.Module):
    """
    split text in chunks of <=512 tokens and combine outputs somehow
    """

    def __init__(self, n_outputs):
        super(BERT_hierarchical_RNN, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              output_attentions = False,
                                              output_hidden_states = True)

        #self.rnn = nn.RNN(768, 264, 2) # TODO: wieviele layers? welche hidden size?
        self.rnn = nn.GRU(768, 264, 2) # auch mal ausprobieren?
        #self.rnn = nn.LSTM(768, 264, 2)

        self.out = nn.Sequential(nn.Linear(529, 128), # 2*264 + 1
                                  nn.LeakyReLU(0.01),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU(0.01),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(64, n_outputs)
                                  )

    def forward(self, section_input_ids, section_attention_mask, textlength):
        nr_sections = section_input_ids.size()[1]
        sections_out = []
        #print("nr_sections in this batch:", nr_sections)
        #print(section_input_ids.size())
        #print(section_attention_mask.size())
        for nr in range(nr_sections):
            #print("nr", nr)
            input_ids = section_input_ids[:,nr,:]
            #print(input_ids.size())
            attention_mask = section_attention_mask[:,nr,:]
            #print(attention_mask.size())
            out_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state_cls = out_bert[1]  # pooled output (hidden state of first token with some modifications)
            #print(hidden_state_cls.size())
            sections_out.append(hidden_state_cls)
        sections_out = torch.stack(sections_out, dim=1) # (batchsize, sections, 768)
        #print(sections_out)
        #print("section_out", sections_out.size())

        # now an rnn over all section_out per section
        #print(sections_out.permute(1,0,2).size())
        output, hidden_state_last_timestep = self.rnn(sections_out.permute(1,0,2))
        #print("output", output.size())
        #print("hidden_state_last_timestep", hidden_state_last_timestep.size())
        hidden_state_last_timestep = hidden_state_last_timestep.permute(1,0,2)
        stacked = hidden_state_last_timestep.reshape(hidden_state_last_timestep.size()[0], -1) # (batch_size, ...)
        #print("stacked", stacked.size())
        concatenated = torch.cat([stacked, textlength], dim=1)
        #print("concat", concatenated.size())

        out = self.out(concatenated)
        #print("out", out.size())



        #section_sum = torch.sum(sections_out, dim=1)
        #print(section_sum)
        #print(section_sum.size())
        #concatenated = torch.cat([section_sum, textlength], dim=1)
        #print(concatenated)
        #print(concatenated.size()) # just the "textlength" sum and the texlength

        #out = self.out(concatenated)

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

