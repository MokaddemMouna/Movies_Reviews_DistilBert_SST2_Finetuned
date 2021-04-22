import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import *
from sklearn.metrics import classification_report
from process_data import *
from train_classifier import max_seq_length, batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder = os.getcwd()
data_file = os.path.join(current_folder, 'test.tsv')

labels = ['negative','positive']

def prepare_data(df):

    labels = list(df.iloc[:,1])
    samples = list(df.iloc[:,0])

    # tokenize data with distilbert tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    encodings = tokenizer(samples, return_tensors='pt' , max_length=max_seq_length, padding=True, truncation=True)
    input_ids = encodings['input_ids']  # tokenized and encoded sentences
    attention_masks = encodings['attention_mask']  # attention masks
    test_labels = torch.tensor(labels)

    # make test loader
    test_data = TensorDataset(input_ids, attention_masks, test_labels)
    test_sampler =  SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    # save test loader
    torch.save(test_dataloader, 'test_data_loader')


def test_and_print_report():

    # load model
    model = DistilBertForSequenceClassification.from_pretrained('finetuned_distilbert_lg').to(device)
    # load test dataloader
    test_dataloader = torch.load('test_data_loader')
    # Put model in evaluation mode
    model.eval()

    # track variables
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from the dataloader
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

    # Print and save classification report
    clf_report = classification_report(true_labels, pred_labels, target_names=labels)
    pickle.dump(clf_report, open('classification_report.txt', 'wb'))  # save report
    print(clf_report)





data = load_data(data_file)
prepare_data(data)
test_and_print_report()
