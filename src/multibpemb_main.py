import argparse
from Util import *
import pandas as pd
from transformers.optimization import *
import logging
import pathlib
from Models import *
from bpemb import BPEmb


def get_default_encoder_args():
    return argparse.Namespace(
        embed_num=64,
        embed_dim=300,
        image_channels_input=1,
        image_channels_output=100,
        kernel_sizes=[1, 3, 6, 9],
        dropout=0.5,
        number_classes=4,
        encoder_layers=2
    )


def get_default_basemodel_args():
    return argparse.Namespace(
        image_channels_input=1,
        image_channels_output=200,
        kernel_sizes=[3, 4, 5, 6, 7, 8],
        dropout=0.7,
        number_classes=4,
        embed_num=64,
        embed_dim=300
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', type=bool, default=False, help='toy evaluation for development purposes')
    parser.add_argument('--model', type=str, default='base', help='the model to use')
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
    parser.add_argument('--voc_size', type=int, default=100000, help='the size of the subword vocabulary')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--val_n_rounds', type=int, default=3, help='validation round every n epochs')
    parser.add_argument('--max_seq_len', type=int, default=64, help='max amount of BERT tokens')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_file', type=str, default=None, help='save file for the trained model')
    parser.add_argument('--chunk_size', type=int, default=1000000, help='chunksize to read file in (due to large file)')

    return parser.parse_args()


if __name__ == '__main__':

    logger = logging.getLogger('multilingual-engagement-prediction')

    args = parse_args()

    multibpemb = BPEmb(lang="multi", vs=args.voc_size, dim=300)

    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = None
    if args.model == 'base':
        model = Base(get_default_basemodel_args(), enbeddings=multibpemb)
    elif args.model == 'encoder':
        model = Encoder(get_default_encoder_args(), enbeddings=multibpemb)
    else:
        raise ValueError(f'Model {args.model} not recognized.')

    if args.load_model_path is not None:
        logger.log(msg=f'Loading model from {args.load_model_path}', level=logging.INFO)
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.log(msg='Started training.', level=logging.INFO)

    for epoch in range(args.epochs):
        train_data = pd.read_csv(args.train_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                                 header=None, usecols=CONTENT_BASED_COLUMNS + LABELS, chunksize=args.chunk_size)



        for chunk in train_data:

            if args.dev_mode:
                chunk = chunk.sample(1000)

            train_dataloader = prepare_multibpemb_dataset(chunk, args.batch_size, encoder=multibpemb,
                                                          max_seq_len=args.max_seq_len)
            train_model(model=model, dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, device=device)

        if args.val_file is not None and epoch % args.val_n_rounds == 0:
            logger.log(msg='Started validation.', level=logging.INFO)
            val_data = pd.read_csv(args.val_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                                   header=None, usecols=CONTENT_BASED_COLUMNS + LABELS)
            val_dataloader = prepare_multibpemb_dataset(val_data, batch_size=args.batch_size,
                                                        encoder=multibpemb)
            test_model(model=model, dataloader=val_dataloader, device=device)
            logger.log(msg='Finished validation.', level=logging.INFO)

    logger.log(msg='Finished training.', level=logging.INFO)

    if args.save_file is not None:
        save_model(model=model, path_to_save_file=pathlib.Path(args.save_file))

    logger.log(msg='Started testing.', level=logging.INFO)
    test_data = pd.read_csv(args.test_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                            header=None, usecols=CONTENT_BASED_COLUMNS + LABELS)
    test_dataloader = prepare_multibpemb_dataset(test_data, args.batch_size, encoder=multibpemb,
                                                 max_seq_len=args.max_seq_len)
    test_model(model=model, dataloader=test_dataloader, device=device)
    logger.log(msg='Finished testing.', level=logging.INFO)
