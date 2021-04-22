import os
import sqlite3
import pandas as pd
from pathlib import Path
import gc
from tqdm import trange
import numpy as np

import torch
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

num_labels = 2
max_seq_length = 300
batch_size = 32
epochs = 10

def load_data():
    conn = sqlite3.connect('./db.sqlite')
    df = pd.read_sql('select * from movie_reviews', conn)
    return df

def data_analysis(df):
    print('Unique comments: ', df.review.nunique() == df.shape[0])

    print('Null values: ', df.isnull().values.any())
    # check average length and std of samples
    print('average sentence length: ', df.review.str.split().str.len().mean())
    print('stdev sentence length: ', df.review.str.split().str.len().std())

    # Label counts, may need to downsample or upsample
    print('Count of 1 in label: \n', df.sentiment.sum(), '\n')
    print('Count of 0 in label: \n', df.sentiment.eq(0).sum())

def prepare_data(df):
    # shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)

    labels = list(df.sentiment.values)
    samples = list(df.review.values)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)  # tokenizer
    # used the depricated version of the method because it outputs lists which I need to do the stratified split
    encodings = tokenizer.batch_encode_plus(samples, max_length=max_seq_length, padding=True, truncation=True)
    input_ids = encodings['input_ids']  # tokenized and encoded sentences
    attention_masks = encodings['attention_mask']  # attention masks

    train_inputs , validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(
        input_ids, labels, attention_masks, random_state=2020, test_size=0.10, stratify=labels)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    torch.save(validation_dataloader, 'validation_data_loader')
    torch.save(train_dataloader, 'train_data_loader')

def train_and_validate():
    # empty cache to allow more memory for gpu
    gc.collect()
    torch.cuda.empty_cache()

    # Load model, the pretrained model will include a single linear classification layer on top for classification.
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=num_labels)
    model.cuda()

    train_dataloader = torch.load('train_data_loader')

    # setting custom optimization parameters.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr= 1e-5, correct_bias=True)

    # Store the loss and accuracy for plotting if needed
    train_loss_set = []

    # Store the F1 max
    f1_max = 0


    # trange is a tqdm wrapper around the normal python range
    for i in trange(epochs, desc="Epoch"):
        # check if F1 has flattened, note here that because of computational capacity limitation, I assume that flatten
        # for me is only to check that for 2 consecutive epochs F1 haven't increased by at least 1 %, otherwise if
        # F1 continues to grow then it will be capped by the # epochs
        # if i > 2:
        #     f1_current = f1_set[-1]
        #     f1_previous = f1_set[-2]
        #     f1_before_previous = f1_set[-3]
        #     diff_curr = f1_current - f1_previous
        #     diff_prev = f1_previous - f1_before_previous
        #     diff = (diff_curr + diff_prev) / 2
        #     if diff < 1 and diff_curr < 1:
        #         break

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0  # running loss
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # Forward pass for binary classification
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            train_loss_set.append(loss.item())

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        ###############################################################################

        # Validation

        validation_dataloader = torch.load('validation_data_loader')

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                outs = model(b_input_ids, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.argmax(b_logit_pred, dim=1)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        f1_macro = f1_score(true_labels, pred_labels, average='macro') * 100
        f1_micro = f1_score(true_labels, pred_labels, average='micro') * 100

        if f1_max < f1_micro:
            print('this is a better model to save')
            model_to_save = model
            f1_max = f1_micro

        print('F1 Validation Score Micro: ', f1_micro)
        print('F1 Validation Score Macro: ', f1_macro)


    # store the best model
    model_to_save.save_pretrained('finetuned_distilbert_lg')






if __name__ == '__main__':
    train, validation = Path('./train_data_loader'), Path('./validation_data_loader')
    finetuned_model = './finetuned_distilbert_lg'
    if not (train.is_file() and validation.is_file()):
        data = load_data()
        data_analysis(data)
        prepare_data(data)
    if not os.path.exists(finetuned_model):
        train_and_validate()


