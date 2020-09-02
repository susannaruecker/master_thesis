import torch
from master_thesis.src import utils, models, data
import scipy.stats as st
import numpy as np



def predict(which,  # must be "bert" or "cnn"
            identifier,  # the identifier to base which model to load (mostly fixed_len)
            val_batch_size = 200,
            on_set = 'dev'
            ):

    print(f"Predicting with model {which.upper()} with identifier {identifier}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load the trained and saved model
    if which == "bert":
        model, tokenizer = models.get_model_and_tokenizer(utils.OUTPUT / 'saved_models' / f'BERT_{str(identifier)}')
        model.to(device)

    if which == "cnn":
        embs_dim = 300
        model = models.CNN(num_outputs=1,
                           embs_dim=embs_dim,
                           filter_sizes=[3, 4, 5],
                           num_filters=[100, 100, 100]
                           )
        model.load_state_dict(torch.load(utils.OUTPUT / 'saved_models' / f'CNN_{str(identifier)}.pt',
                                         map_location = torch.device(device)))
        model.to(device)
        # load embeddings
        embs = utils.load_fasttext_vectors(limit=None)

    # get data (already conditionend on min_pageviews etc)
    df = utils.get_conditioned_df()
    df = df[['text_preprocessed', 'avgTimeOnPagePerNr_tokens']] # to save space?


    # load Dataloader for dev-Set (batch size in interference can be big, no calculation needed)
    if which == "bert":
        MIN_LEN = 400
        START = None
        FIXED_LEN = identifier

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

    if which == "cnn":
        MIN_LEN = 0 #None # 400
        START = 0 #None
        FIXED_LEN = identifier

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
            if which == "bert":
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                pred_dev = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            if which == "cnn":
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
    #predict(which="bert", identifier=None, val_batch_size=200, on_set='dev') # hier wird wieder immer der gleiche Wert gegeben... optimizer geändert
    predict(which="cnn", identifier=500, val_batch_size=200, on_set='dev')




### BERT Evaluation
# bei max_len 100 (0.46954919345639684, 7.58807236498442e-71) also ähnlich wie BOW
# bei max_len 200 (0.5467132606724006, 3.380111928800224e-100) also ziemlich eindeutig besser
# bei max_len 300 (0.5927131137308533, 1.0772004471333138e-121)
# bei max_len 512 (0.629376173357152, 1.727308917370859e-141)
# bei FIXED_LEN: None, MIN_LEN: 400, START: None (0.6714032213951131, 9.993981267919477e-168)
# bei FIXED_LEN None, MIN_LEN 500 BATCH_SIZE 6, START None, lr 1e-5: (0.6670378344980564, 8.423550985343239e-165)
# BERT_FIXLENNone_MINLEN500_STARTNone_EP5_BS6_LR1e-06     (0.6675789223620894, 3.6775100957520147e-165)


### CNN Evaluation
# bei max_len = 200: (0.5574401705693174, 6.52347471105848e-105) (EPOCHS = 3)
# bei max_len = 400: (0.6242538472164617, 1.4538840723780503e-138) (EPOCHS = 2)
# bei max_len = 300 und neu average statt max_pooling: (0.6630899615610468, 3.3797950507796264e-162)
# bei max_len = 500 und neu average statt max_pooling (0.6666460255912416, 1.533416216477133e-164)
# bei FIXED_LEN: None, MIN_LEN: 400, START: None (0.6287055687453889, 4.201947567836372e-141)
# bei FIXED_LEN None, MIN_LEN 500, START None, Epochs 8, LR 1e-4 (0.6308993289053406, 2.2746886605639486e-142)

