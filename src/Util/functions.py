import torch
from sklearn.metrics import precision_recall_curve, auc, log_loss
import pandas as pd
from .constant_objs import *
import logging
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
import transformers

logger = logging.getLogger('multilingual-engagement-prediction')


def save_model(model, path_to_save_file):
    torch.save(model.state_dict(), path_to_save_file)


def load_model(model_cls, args, path_to_model):
    loaded_model = model_cls(args)
    loaded_model.load_state_dict(torch.load(path_to_model))

    return loaded_model


def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)

    data_ctr = calculate_ctr(gt)

    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])

    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def compute_all_metrics(pred, gt):
    print(f'Number instances: {len(pred)} PR-AUC: {compute_prauc(pred, gt)} RCE: {compute_rce(pred, gt)}')


def handle_label(x):
    #  labels that are false are None in dataset
    if pd.isnull(x):
        return 0
    else:
        return 1


def transform_labels(data):
    for lab in LABELS:
        data[lab] = data[lab].apply(lambda x: handle_label(x))
        logger.log(msg=f'Parsed label {lab}, value counts are {data[lab].value_counts()}', level=logging.INFO)

def _decode_tweet_text(x, tokenizer):
    return "".join(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x.split('\t')))) \
        .replace('[CLS]', '').replace('[SEP]', '')


def _encode_sentence(x, tokenizer, max_seq_len):

    encoded_dict = tokenizer.encode_plus(
        x,
        add_special_tokens=True,
        max_length=max_seq_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True
    )

    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def prepare_dataset(data, max_seq_len, tokenizer):
    # bert tokenizer needed as text passed as BERT IDs in original dataset
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    decoded_text = data['text_tokens'].apply(lambda x: _decode_tweet_text(x, bert_tokenizer))

    ids_list = []
    mask_list = []
    for s in iter(decoded_text.to_list()):
        ids, mask = _encode_sentence(s, tokenizer=tokenizer, max_seq_len=max_seq_len)
        ids_list.append(ids)
        mask_list.append(mask)

    #s = decoded_text.apply(lambda x: _encode_sentence(x, tokenizer, max_seq_len))
    input_ids = torch.tensor(ids_list)
    attention_masks = torch.tensor(mask_list)
    labels = torch.tensor(data[LABELS].values.tolist())

    return TensorDataset(input_ids, attention_masks, labels)


def train_sota_model(model, dataloader, epochs, device, criterion, optimizer, scheduler=None):
    for epoch in range(epochs):

        total_train_loss = 0
        # set the model in train mode
        model.train()

        for batch in tqdm(dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            logits = model(b_input_ids.long(),
                           #token_type_ids=None,
                           attention_mask=b_input_mask.long())

            loss = criterion(logits[0], b_labels.long())
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()


def train_model(model, dataloader, device, criterion, optimizer):
        total_train_loss = 0
        # set the model in train mode
        model.train()

        for batch in tqdm(dataloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            model.zero_grad()
            logits = model(inputs.long())
            loss = criterion(logits, labels.float())
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

def test_model(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions, true_labels = [], []

    for batch in tqdm(dataloader):
        inputs, labels = batch
        labels = labels.to(device)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs.long())

        logits = outputs.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    _compute_all_metrics_multilabel(predictions, true_labels)


def _compute_all_metrics_multilabel(pred, gt):
    pred_frame = pd.DataFrame(pred, columns=LABELS)
    gt_frame = pd.DataFrame(gt, columns=LABELS)

    for label in LABELS:
        pred = pred_frame[label]
        gt = gt_frame[label]
        print(f'Metrics for label {label}: PR-AUC: {compute_prauc(pred, gt)} RCE: {compute_rce(pred, gt)}')


def test_sota_model(model, dataloader, device):
    model.eval()
    model.to(device)
    predictions, true_labels = [], []

    for batch in tqdm(dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.to(device)
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids.long(),
                            #token_type_ids=None,
                            attention_mask=b_input_mask.long())

        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    _compute_all_metrics_multilabel(predictions, true_labels)


def _encode_and_pad(text, maximum_length, encoder=None, padding_token_id=None, convert_empty_to_None=False):
    number_empty_text_rows = 0
    encoded_ids_text = encoder.encode_ids(text)[0:63]

    if len(encoded_ids_text) == 0:
        number_empty_text_rows = number_empty_text_rows + 1
        if convert_empty_to_None:
            return None

    if padding_token_id is not None:
        encoded_ids_text = np.pad(encoded_ids_text, pad_width=(0, maximum_length - len(encoded_ids_text)),
                                  mode='constant', constant_values=padding_token_id)
    return encoded_ids_text


def prepare_multibpemb_dataset(dataset, batch_size, encoder, max_seq_len):
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    decoded_tweet_text = dataset['text_tokens'].apply(lambda x: _decode_tweet_text(x, bert_tokenizer))

    transform_labels(dataset)

    ids = decoded_tweet_text.apply(
        lambda text: _encode_and_pad(text, max_seq_len, padding_token_id=0, encoder=encoder))
    ids = ids.apply(lambda ids: torch.tensor(ids).long())
    ds = TensorDataset(torch.stack(ids.tolist(), dim=0),
                            torch.tensor(dataset[LABELS].values.tolist()))

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)


def prepare_char_dataset(dataset, batch_size, max_length=512, padding_unicode_id=1):
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    decoded_tweet_text = dataset['text_tokens'].apply(lambda text_tokens: "".join(
        bert_tokenizer.convert_tokens_to_string(
            bert_tokenizer.convert_ids_to_tokens([int(token) for token in text_tokens.split('\t')]))).replace('[CLS]',
                                                                                                              '').replace(
        '[SEP]', ''))

    transform_labels(dataset)

    ids = decoded_tweet_text.apply(
        lambda text: (([int(ord(char)) for char in list(text)] + [int(padding_unicode_id)] * (max_length - len(text)))[
                      0:max_length]))
    ids = ids.apply(lambda ids: torch.tensor(ids).long())
    dataset = TensorDataset(torch.stack(ids.tolist(), dim=0),
                            torch.tensor(dataset[LABELS].values.tolist()))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)