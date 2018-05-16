from pathlib import Path
import logging
import pickle


def save_file_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_file_pickle(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
        return obj


def read_training_files(data_path):
    """
    Read data from directory
    """
    PATH = Path(data_path)

    with open(PATH/'train.function', 'r') as f:
        t_enc = f.readlines()

    with open(PATH/'valid.function', 'r') as f:
        v_enc = f.readlines()

    # combine train and validation and let keras split it randomly for you
    tv_enc = t_enc + v_enc

    with open(PATH/'test.function', 'r') as f:
        h_enc = f.readlines()

    with open(PATH/'train.docstring', 'r') as f:
        t_dec = f.readlines()

    with open(PATH/'valid.docstring', 'r') as f:
        v_dec = f.readlines()

    # combine train and validation and let keras split it randomly for you
    tv_dec = t_dec + v_dec

    with open(PATH/'test.docstring', 'r') as f:
        h_dec = f.readlines()

    logging.warning(f'Num rows for encoder training + validation input: {len(tv_enc):,}')
    logging.warning(f'Num rows for encoder holdout input: {len(h_enc):,}')

    logging.warning(f'Num rows for decoder training + validation input: {len(tv_dec):,}')
    logging.warning(f'Num rows for decoder holdout input: {len(h_dec):,}')

    return tv_enc, h_enc, tv_dec, h_dec
