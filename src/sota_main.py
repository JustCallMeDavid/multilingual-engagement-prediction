import argparse
from Util import *
import pandas as pd
from transformers.optimization import *
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, \
    XLMRobertaForSequenceClassification, XLMForSequenceClassification, AdamW, BertTokenizer, XLMRobertaTokenizer, \
    XLMTokenizer, DistilBertTokenizer
import logging
import pathlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', type=bool, default=False, help='toy evaluation for development purposes')
    parser.add_argument('--model', type=str, default='bert', help='the model to use')
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--max_seq_len', type=int, default=64, help='max amount of BERT tokens')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--save_file', type=str, default=None, help='save file for the trained model')

    return parser.parse_args()


if __name__ == '__main__':

    logger = logging.getLogger('multilingual-engagement-prediction')

    args = parse_args()

    train_data = pd.read_csv(args.train_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                             header=None, dtype = object)
    test_data = pd.read_csv(args.test_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                            header=None, dtype = object)

    if args.dev_mode:
        train_data = train_data.sample(1000)
        test_data = test_data.sample(1000)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform_labels(train_data)
    transform_labels(test_data)

    model = None
    tokenizer = None
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)
    elif args.model == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased', do_lower_case=False)
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=4)
    elif args.model == 'roberta':
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=False)
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=4)
    elif args.model == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048', do_lower_case=False)
        model = XLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048', num_labels=4)
    else:
        raise ValueError(f"Model {args.model} not valid.")

    logger.log(msg = 'Preparing train data', level=logging.INFO)
    train_dataset = prepare_dataset(train_data, max_seq_len=args.max_seq_len, tokenizer=tokenizer)
    logger.log(msg = 'Preparing test data.', level=logging.INFO)
    test_dataset = prepare_dataset(test_data, max_seq_len=args.max_seq_len, tokenizer=tokenizer)
    logger.log(msg= 'Data preparation completed.', level=logging.INFO)

    if args.load_model_path is not None:
        logger.log(msg=f'Loading model from {args.load_model_path}', level=logging.INFO)
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    criterion = torch.nn.BCEWithLogitsLoss()

    logger.log(msg='Started training.', level=logging.INFO)
    train_sota_model(model=model, dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, epochs=args.epochs,
                     scheduler=scheduler, device=device)
    logger.log(msg='Finished training.', level=logging.INFO)

    if args.save_file is not None:
        save_model(model=model, path_to_save_file=pathlib.Path(args.save_file))

    test_sota_model(model=model, dataloader=test_dataloader, device=device)
