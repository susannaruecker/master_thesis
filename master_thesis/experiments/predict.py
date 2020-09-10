import torch
from master_thesis.src import utils, models, data
from transformers import BertTokenizer, BertForSequenceClassification
import scipy.stats as st
import numpy as np

#TODO: Das hier ist alles noch nicht so perfekt, weil ich so oft die konkrete Art/Namen/Speichermethoden der Modelle verändert habe...

def predict(which,  # must be "bert" or "cnn"
            identifier,  #
            val_batch_size = 200,
            on_set = 'dev'
            ):

    print(f"Predicting with model {identifier}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load the trained and saved model
    PRE_TRAINED_MODEL_NAME = 'bert-base-german-cased'
    if which == "BERT":

        # für die Modelle, die noch mit der "alten Struktur" trainiert wurden ist möglicherweise nötig:
        model = BertForSequenceClassification.from_pretrained(utils.OUTPUT / 'saved_models' / f'{str(identifier)}',
                                                              num_labels = 1, # turns "classification" into regression?
                                                              output_attentions = False,
                                                              output_hidden_states = False,
                                                              )

        #model = models.Bert_sequence(n_outputs=1)
        #model.load_state_dict(torch.load(utils.OUTPUT / 'saved_models' / f'{str(identifier)}',
        #                                 map_location = torch.device(device)))
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model.to(device)

    if which == "BERTModel":
        model = models.Bert_regression(n_outputs=1)
        model.load_state_dict(torch.load(utils.OUTPUT / 'saved_models' / f'{str(identifier)}',
                                      map_location=torch.device(device)))
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model.to(device)

    if which == "CNN":
        embs_dim = 300
        #model = models.CNN(num_outputs=1,
        #                   embs_dim=embs_dim,
        #                   filter_sizes=[3, 4, 5],
        #                   num_filters=[100, 100, 100]
        #                   )
        model = models.CNN_small(num_outputs=1,
                                 embs_dim=embs_dim,
                                 filter_sizes=[3, 4, 5],
                                 num_filters=[20, 20, 20])

        model.load_state_dict(torch.load(utils.OUTPUT / 'saved_models' / f'{str(identifier)}',
                                         map_location = torch.device(device)))
        model.to(device)
        # load embeddings
        embs = utils.load_fasttext_vectors(limit=None)

    # get data (already conditionend on min_pageviews etc)
    df = utils.get_conditioned_df()
    df = df[['text_preprocessed', 'avgTimeOnPagePerNr_tokens']] # to save space?


    # load Dataloader for dev-Set (batch size in interference can be big, no calculation needed)
    if which == "BERT" or which == "BERTModel":
        FIXED_LEN = None
        MIN_LEN = 500
        START = None

        window = data.RandomWindow_BERT(start=START, fixed_len=FIXED_LEN, min_len=MIN_LEN)
        collater = data.Collater_BERT()

        dl_train, dl_dev, dl_test = data.create_DataLoaders_BERT(df = df,
                                                                 target = 'avgTimeOnPagePerNr_tokens',
                                                                 text_base = 'text_preprocessed',
                                                                 tokenizer = tokenizer,
                                                                 val_batch_size= val_batch_size,
                                                                 train_batch_size=val_batch_size, # here okay because just for inference
                                                                 collater = collater,
                                                                 transform = window)

    if which == "CNN":
        FIXED_LEN = None
        MIN_LEN = 500
        START = None

        window = data.RandomWindow_CNN(start=START, fixed_len=FIXED_LEN, min_len=MIN_LEN)
        collater = data.Collater_CNN()

        dl_train, dl_dev, dl_test = data.create_DataLoaders_CNN(df=df,
                                                                target='avgTimeOnPagePerNr_tokens',
                                                                text_base='text_preprocessed',
                                                                tokenizer=None,  # uses default (spacy) tokenizer
                                                                embs=embs,
                                                                train_batch_size=val_batch_size, # here okay because just for inference
                                                                val_batch_size=val_batch_size,
                                                                transform=window,
                                                                collater=collater)

    if on_set == 'train':
        dl = dl_train
    if on_set == 'dev':
        dl = dl_dev
    if on_set == 'test':
        dl = dl_test

    pred = []
    true = []

    model.eval()
    with torch.no_grad():
        for d in dl:
            if on_set == 'train' and len(pred) > 2000: # don't use whole train, takes a long time
                break
            if which == "BERT" or which == "BERTModel":
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                pred_dev = model(input_ids=input_ids, attention_mask=attention_mask) #[0] # das muss leider bei den "alten" hin, neu nicht mehr
            if which == "CNN":
                input_matrix = d["input_matrix"].to(device)
                pred_dev = model(input_matrix)

            #print(pred_dev[:10])
            y_dev = d["target"].to(device)

            pred_dev = np.array(pred_dev.squeeze().cpu())
            y_dev = np.array(y_dev.squeeze().cpu())

            print("Pearson's r of this batch:", st.pearsonr(pred_dev, y_dev))

            pred.extend(pred_dev)
            true.extend(y_dev)

    print("Pearson's r on whole set:", st.pearsonr(pred, true))
    return np.array(pred), np.array(true)



if __name__ == "__main__":
    ## ACHTUNG, etwas unschön:
    ## Man muss bisher händisch die Hyperparameter eingeben und auch (bei CNN vor allem), das richtige Model auswählen
    #predict(which="CNN", identifier="CNN_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR0.0001_smaller", on_set='dev')
    #predict(which="BERT", identifier="BERT_FIXLENNone_MINLEN500_STARTNone_EP7_BS8_LR1e-05", on_set='dev')
    predict(which="BERTModel", identifier="BERTModel_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR1e-05", on_set='dev')



### BERT Evaluation
###
# bei max_len 100 (0.469) also ähnlich wie BOW
# bei max_len 200 (0.547) also ziemlich eindeutig besser
# bei max_len 300 (0.593)
# bei max_len 512 (0.629)
# bei FIXED_LEN: None, MIN_LEN: 400, START: None (0.671)
# bei FIXED_LEN None, MIN_LEN 500 BATCH_SIZE 6, START None, lr 1e-5: (0.667)
# BERT_FIXLENNone_MINLEN500_STARTNone_EP5_BS6_LR1e-06     (0.668)
# BERT_FIXLENNone_MINLEN500_STARTNone_EP10_BS8_LR1e-06    (0.656)
    # das ist nach 6 Epochenstark wieder gesunken, siehe Tensorboard
# BERT_FIXLENNone_MINLEN500_STARTNone_EP7_BS8_LR1e-05     (0.613)
    # das war am Anfang seeeehr gut (0.67), ist dann aber massiv gesunken --> Tensorboard
    # Idee: unbedingt mal Dropout erhöhen
# BERT_FIXLENNone_MINLEN500_STARTNone_EP2_BS8_LR1e-06    (0.652) (das war nur zum Testen nach Code-Umbau)

### BERTModel (also nicht BertForSequenceClassification sondern eigene Architektur) mit mehr Dropout
# BERTModel_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR1e-05 (0.627) # erste sehr gut (0.68), dann abgefallen
# BERTModel_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR1e-06 (0.673)
    # das war ohne das zusätzliche Linear Layer
# noch laufen lassen? BERTModel_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR0.0001

### BERTAvg (also gemittelte last hdden states statt CLS-Token, eigenes Dropout)
# BERTAvg_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR1e-05 (0.634)
    # das war ganz am Anfang sehr gut, ist dann aber sofort abgefallen

### mit gradient accumulation (nur jede 5. batch optimizer)
# BERT_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR1e-05_gradient_acc (0.659) # war vorher etwas besser, dann wieder gesunken
# BERTModel_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR1e-05_gradient_acc (0.673) # leicht wieder gefallen
# BERTAvg_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR1e-05_gradient_acc (0.6702) # leicht wieder gefallen
# BERTModel_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR1e-06_gradient_acc (0.656)
# BERT_FIXLENNone_MINLEN500_STARTNone_EP3_BS8_LR0.0001_gradient_acc (0.651)
# BERTAvg_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR1e-06_gradient_acc (0.663)


### CNN Evaluation
###
# bei max_len = 200: (0.557) (EPOCHS = 3)
# bei max_len = 400: (0.624) (EPOCHS = 2)
# bei max_len = 300 und neu average statt max_pooling: (0.663)
# bei max_len = 500 und neu average statt max_pooling (0.667) # war hier lr=0.001?
# bei FIXED_LEN: None, MIN_LEN: 400, START: None (0.629)
# bei FIXED_LEN None, MIN_LEN 500, START None, Epochs 8, LR 1e-4 (0.631)
# CNN_FIXLENNone_MINLEN500_STARTNone_EP10_BS8_LR0.001 (0.632)
# CNN_FIXLENNone_MINLEN500_STARTNone_EP10_BS8_LR0.0001 (0.634)


# Adadelta ist schlecht, lr = 1e-5 auch
# was ist mit Adam statt AdamW?


### CNN_smaller (30,30,30)
# CNN_FIXLENNone_MINLEN500_STARTNone_EP3_BS8_LR0.0001_smaller (0.635)

### CNN_smaller (50,50,50), und mit weniger Dropout (0.3)
# CNN_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR0.0001_smaller (0.63)

# even smaller:
# CNN_FIXLENNone_MINLEN500_STARTNone_EP5_BS8_LR0.0001_smaller (0.635)
# CNN_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR0.0001_smaller_newFilter (0.629)
# CNN_FIXLENNone_MINLEN500_STARTNone_EP4_BS8_LR0.0001_smaller_newFilter (0.632) # Dropout kleiner (0.3)

### (60,60,60), 0.4 Dropout
# CNN_FIXLENNone_MINLEN500_STARTNone_EP6_BS8_LR0.0001_smaller (0.633)

### (10,10,10), 0.4 Dropout
# CNN_FIXLENNone_MINLEN500_STARTNone_EP3_BS8_LR0.0001_very_small (0.628)



