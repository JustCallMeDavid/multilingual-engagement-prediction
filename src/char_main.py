import argparse
from Util import *
import pandas as pd
from transformers.optimization import *
import logging
import pathlib
from Models import *


def get_default_char_args():
    return argparse.Namespace(
        emb_num=512,
        emb_dim=100,
        ch_in=1,
        ch_out=256,
        kernel_size_pool=(2, 1),
        stride_pool=(2, 1),
        number_convolutions=3,
        forward_layer_sizes=[1024, 256, 32],
        dropout=0.1,
        number_classes=4,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_mode', type=bool, default=False, help='toy evaluation for development purposes')
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
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

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = Char(get_default_char_args())

    if args.load_model_path is not None:
        logger.log(msg=f'Loading model from {args.load_model_path}', level=logging.INFO)
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):

        train_data = pd.read_csv(args.train_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                                 header=None,
                                 usecols=CONTENT_BASED_COLUMNS + LABELS, chunksize=args.chunk_size)
        for chunk in train_data:

            if args.dev_mode:
                chunk = chunk.sample(1000)

            train_dataloader = prepare_char_dataset(dataset=chunk,
                                                    batch_size=args.batch_size,
                                                    max_length=args.max_seq_len,
                                                    padding_unicode_id=1)

            train_model(model=model, dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, device=device)

        if args.val_file is not None and epoch % args.val_n_rounds == 0:
            logger.log(msg='Started validation.', level=logging.INFO)
            val_data = pd.read_csv(args.val_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                                   header=None, usecols=CONTENT_BASED_COLUMNS + LABELS)
            val_dataloader = prepare_char_dataset(val_data, batch_size=args.batch_size,
                                                  max_length=args.max_seq_len)
            test_model(model=model, dataloader=val_dataloader, device=device)
            logger.log(msg='Finished validation.', level=logging.INFO)

    logger.log(msg='Finished training.', level=logging.INFO)

    if args.save_file is not None:
        save_model(model=model, path_to_save_file=pathlib.Path(args.save_file))

    logger.log(msg='Started testing.', level=logging.INFO)
    test_data = pd.read_csv(args.test_file, sep='\x01', encoding='utf-8', names=CONTENT_BASED_COLUMNS + LABELS,
                            header=None, usecols=CONTENT_BASED_COLUMNS + LABELS)
    test_dataloader = prepare_char_dataset(test_data, batch_size=args.batch_size,
                                           max_length=args.max_seq_len)
    test_model(model=model, dataloader=test_dataloader, device=device)
    logger.log(msg='Finished testing.', level=logging.INFO)