import torch
from master_thesis.src import utils, models, data
import scipy.stats as st
import numpy as np


def predict(which,      # must be "bert" or "cnn"
            max_len,    # the max_len the model was trained on
            batch_size = 200,
            ):

    print(f"Predicting with model {which.upper()} trained on max_len {max_len}...")
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    print("Device is: ", device)

    # load the trained and saved model
    if which == "bert":
        model, tokenizer = models.get_model_and_tokenizer(utils.OUTPUT / 'saved_models' / f'BERT_{str(max_len)}')
        model.to(device)

    if which == "cnn":
        embs_dim = 300
        model = models.CNN(num_outputs=1,
                           embs_dim=embs_dim,
                           filter_sizes=[3, 4, 5],
                           num_filters=[100, 100, 100]
                           )
        model.load_state_dict(torch.load(utils.OUTPUT / 'saved_models' / f'CNN_{str(max_len)}.pt'))
        model.to(device)
        # load embeddings
        embs = utils.load_fasttext_vectors(limit=None)

    # get data (already conditionend on min_pageviews etc)
    df = utils.get_conditioned_df()


    # load Dataloader for dev-Set (batch size in interference can be big, no calculation needed)
    if which == "bert":
        _, dl_dev, _ = data.create_DataLoaders_BERT(df = df,
                                                    target = 'avgTimeOnPagePerNr_tokens',
                                                    text_base = 'text_preprocessed',
                                                    tokenizer = tokenizer,
                                                    max_len = max_len,
                                                    batch_size = batch_size)
    if which == "cnn":
        _, dl_dev, _ = data.create_DataLoaders_CNN(df=df,
                                                   target='avgTimeOnPagePerNr_tokens',
                                                   text_base='text_preprocessed',
                                                   tokenizer=None,
                                                   max_len=max_len,
                                                   batch_size=batch_size,
                                                   embs=embs)


    pred = []
    true = []

    model.eval()
    with torch.no_grad():
        for d in dl_dev:
            if which == "bert":
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                pred_dev = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            if which == "cnn":
                input_matrix = d["input_matrix"].to(device)
                pred_dev = model(input_matrix)

            y_dev = d["target"].to(device)

            pred_dev = np.array(pred_dev.squeeze().cpu())
            y_dev = np.array(y_dev.squeeze().cpu())

            print("Pearson's r of this batch:", st.pearsonr(pred_dev, y_dev))

            pred.extend(pred_dev)
            true.extend(y_dev)

    print("Pearson's r on whole dev set:", st.pearsonr(pred, true))
    return np.array(pred), np.array(true)



if __name__ == "__main__":
    predict(which="bert", max_len=512, batch_size=200)
    #predict(which="cnn", max_len=400, batch_size=200)




### BERT Evaluation
# bei max_len 100 (0.46954919345639684, 7.58807236498442e-71) also ähnlich wie BOW
# bei max_len 200 (0.5467132606724006, 3.380111928800224e-100) also ziemlich eindeutig besser
# bei max_len 300 (0.5927131137308533, 1.0772004471333138e-121)
# bei max_len 512 (0.629376173357152, 1.727308917370859e-141)

### CNN Evaluation
# bei max_len = 200: (0.5574401705693174, 6.52347471105848e-105) (EPOCHS = 3)
# bei max_len = 400: (0.6242538472164617, 1.4538840723780503e-138) (EPOCHS = 2)

