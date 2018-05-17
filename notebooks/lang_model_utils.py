from fastai.text import *
from pathlib import Path
from tqdm import tqdm_notebook
from keras.preprocessing.sequence import pad_sequences
from typing import List, Any
from shutil import copyfile
import torch
import logging


def list_flatten(l: List[List[Any]]) -> List[Any]:
    "List[List] --> List"
    return [item for sublist in l for item in sublist]


def _dd():
    "Helper function for defaultdict."
    return 1


def load_lm_vocab(lm_vocab_file: str):
    """load vm_vocab object."""
    with open(lm_vocab_file, 'rb') as f:
        info = pickle.load(f)

    v = lm_vocab()
    v.itos = info['itos']
    v.stoi = info['stoi']
    v.vocab_size = len(v.itos)
    v.max_vocab = info['max_vocab']
    v.min_freq = info['min_freq']
    v.bos_token = info['bos_token']
    logging.warning(f'Loaded vocab of size {v.vocab_size:,}')
    return v


class lm_vocab:
    def __init__(self,
                 max_vocab: int = 50000,
                 min_freq: int = 15,
                 bos_token: str = '_xbos_'):
        """
        Builds vocabulary and indexes string for FastAI language model.

        Parameters
        ==========
        max_vocab : int
            Maximum sie of vocabulary.

        min_freq : int
            Minimum frequency threshold for token to be included in vocabulary.

        bos_token : str
            Beginning of string token
        """
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.bos_token = '_xbos_'

    def fit(self, data: List) -> None:
        "Fit vocabulary to a list of documents."
        logging.warning(f'Processing {len(data):,} rows')
        # build vocab
        trn = list_flatten([(self.bos_token + ' ' + x).split() for x in data])
        freq = Counter(trn)
        itos = [o for o, c in freq.most_common(self.max_vocab) if c > self.min_freq]

        # insert placeholder tokens
        itos.insert(0, '_pad_')
        itos.insert(0, '_unk_')
        self.vocab_size = len(itos)
        logging.warning(f'Vocab Size {self.vocab_size:,}')

        # build reverse index
        stoi = collections.defaultdict(_dd, {v: k for k, v in enumerate(itos)})

        # save vocabulary
        self.itos = dict(enumerate(itos))
        self.stoi = stoi

    def transform_flattened(self, data: List[str]) -> List[int]:
        """Tokenizes, indexes and flattens list of strings for fastai language model."""
        logging.warning(f'Transforming {len(data):,} rows')
        tok_trn = list_flatten([(self.bos_token + ' ' + x).split() for x in data])
        return np.array([self.stoi[s] for s in tok_trn])

    def fit_transform_flattened(self, data: List[str]) -> List[int]:
        "Applies `fit` then `transform_flattened` methods sequentially."
        self.fit(data)
        return self.transform_flattened(data)

    def transfom(self, data: List[str], max_seq_len: int = 60) -> List[List[int]]:
        """Tokenizes, and indexes list of strings without flattening.

        Parameters
        ==========
        data : List[str]
            List of documents (sentences) that you want to transform.
        max_seq_len : int
            The maximum length of any sequence allowed.  Sequences will be truncated
            and pre-padded to this length.
        """
        logging.warning(f'Processing {len(data):,} rows')
        idx_docs = [[self.stoi[word] for word in sent.split()[:max_seq_len]] for sent in data]
        # default keras pad_sequences pre-pads to max length with zero
        return pad_sequences(idx_docs)

    def save(self, destination_file: str) -> None:
        dest = Path(destination_file)
        info = {'stoi': self.stoi,
                'itos': self.itos,
                'max_vocab': self.max_vocab,
                'min_freq': self.min_freq,
                'bos_token': self.bos_token}
        with open(dest, 'wb') as f:
            pickle.dump(info, f)
        logging.warning(f'Saved vocab to {str(dest)}')



def train_lang_model(model_path: int,
                     trn_indexed: List[int],
                     val_indexed: List[int],
                     vocab_size: int,
                     n_cycle: int = 2,
                     em_sz: int = 1200,
                     nh: int = 1200,
                     nl: int = 3,
                     bptt: int = 20,
                     wd: int = 1e-7,
                     bs: int = 32):
    """
    Train fast.ai language model.

    Parameters
    ==========
    model_path : str
        Path where you want to save model artifacts.
    trn_indexed : List[int]
        flattened training data indexed
    val_indexed : List[int]
        flattened validation data indexed
    vocab_size : int
        size of vocab
    n_cycle : int
        Number of cycles to train model.
    em_sz : int
        Word embedding size.
    nh : int
        Dimension of hidden units in RNN
    nl : int
        Number of RNN layers
    bptt : int
        Sequence length for back-propigation through time.
    wd : int
        Weight decay
    bs : int
        Batch size


    Returns
    =======
    Tuple(fastai.learner, pytorch.model)

    Also saves best model weights in file `langmodel_best.torch`
    """
    mpath = Path(model_path)
    mpath.mkdir(exist_ok=True)

    # create data loaders
    trn_dl = LanguageModelLoader(trn_indexed, bs, bptt)
    val_dl = LanguageModelLoader(val_indexed, bs, bptt)

    # create lang model data
    md = LanguageModelData(mpath, 1, vocab_size, trn_dl, val_dl, bs=bs, bptt=bptt)

    # build learner
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7

    learner = md.get_model(opt_fn, em_sz, nh, nl,
                           dropouti=drops[0],
                           dropout=drops[1],
                           wdrop=drops[2],
                           dropoute=drops[3],
                           dropouth=drops[4])

    # learning rate is hardcoded, I already ran learning rate finder on this problem.
    lrs = 1e-3 / 2

    # borrowed these parameters from fastai
    learner.fit(lrs, 2, wds=wd, cycle_len=3, use_clr=(32, 10), best_save_name='langmodel_best')

    # eval sets model to inference mode (turns off dropout, etc.)
    model = learner.model.eval()
    # defensively calling reset to clear model state (b/c its a stateful model)
    model.reset()

    state_dict_dest = mpath/'models/langmodel_best.h5'
    logging.warning(f'State dict for the best model saved here:\n{str(state_dict_dest)}')
    return learner, model


def get_nonzero_indicator_arr(idx_arr, dim):
    """
    Convert the array of indices of size (bs, seq_len) to an indicator
    matrix of size (bs, seq_len, dim) where the 3rd dimension is just
    a copy of the 2nd dimension.  The indicator matrix has values 1 or 0
    indicating 0 for padding and 1 for a non-padding element.  This is
    useful for ingoring padding elements when computing the average.
    """
    assert idx_arr.ndim == 2, 'Input array must be 2D.'
    x = (np.repeat(idx_arr[:, :, np.newaxis], dim, axis=2) != 0)
    # set last item in sequence to True, incase its all zeros
    x[:, -1, :] = True
    return x


def get_mean_emb(raw_emb, idx_arr):
    """
    Get mean hidden state over timesteps ignoring padded elements.
    """
    assert raw_emb.ndim == 3, 'Embedding must have 3 dimensions: (bs, seq_len, dim)'
    nzi = get_nonzero_indicator_arr(idx_arr, dim=raw_emb.shape[-1])
    return np.average(raw_emb, axis=1, weights=nzi)


def get_emb_batch(lang_model, np_array, bs=100):
    """
    Get encoder embeddings from language model in batch.

    Parameters
    ==========
    lang_model : fastai language model
    np_array : numpy.array
        This is an array of shape (bs, seq_len) where each value is an embedding
        index.

    Returns
    =======
    Tuple : (mean_emb, last_emb)

    mean_emb - this is the average of hidden states over time steps excluding padding.
    last_emb - this is the hidden state at the last time step.

    """
    lang_model.eval()
    mean_emb = []
    last_emb = []
    chunksize = np_array.shape[0] // bs
    logging.warning(f'Splitting data into {chunksize} chunks.')
    data_chunked = np.array_split(np_array, chunksize)
    for i in tqdm_notebook(range(len(data_chunked))):
        # get batch
        x = V(data_chunked[i])

        # get raw predictions of shape (bs, seq_len, encoder_dim)
        lang_model.reset()
        y = lang_model(x)[-1][-1].data.cpu().numpy()

        # take the mean of all timesteps, ignoring padding
        # will be of shape (bs, encoder_dim)
        y_mean = get_mean_emb(raw_emb=y, idx_arr=x.data.cpu().numpy())
        # get the last hidden state in the sequence.  Returns arr of size (bs, encoder_dim)
        y_last = y[:, -1, :]

        # collect predictions
        mean_emb.append(y_mean)
        last_emb.append(y_last)

    return np.concatenate(mean_emb), np.concatenate(last_emb)


def list2arr(l: List[int]):
    "Convert list into pytorch Variable."
    raise NotImplementedError

    return V(np.expand_dims(np.array(l), -1)).cpu()


def make_prediction_from_list(model, l):
    """
    Encode a list of integers that represent a sequence of tokens.  The
    purpose is to encode a sentence or phrase.

    Parameters
    -----------
    model : fastai language model
    l : list
        list of integers, representing a sequence of tokens that you want to encode

    """
    raise NotImplementedError
    arr = list2arr(l)# turn list into pytorch Variable with bs=1
    model.reset()  # language model is stateful, so you must reset upon each prediction
    hidden_states = model(arr)[-1][-1] # RNN Hidden Layer output is last output, and only need the last layer

    #return avg-pooling, max-pooling, and last hidden state
    return hidden_states.mean(0), hidden_states.max(0)[0], hidden_states[-1]


def get_embeddings(lm_model, list_list_int):
    """
    Vectorize a list of sequences List[List[int]] using a fast.ai language model.

    Paramters
    ---------
    lm_model : fastai language model
    list_list_int : List[List[int]]
        A list of sequences to encode

    Returns
    -------
    tuple: (avg, mean, last)
        A tuple that returns the average-pooling, max-pooling over time steps as well as the last time step.
    """
    raise NotImplementedError
    avg_embs, mean_embs, last_embs = [], [], []

    for i in tqdm_notebook(range(len(list_list_int))):
        avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])
        avg_embs.append(avg_)
        mean_embs.append(max_)
        last_embs.append(last_)

    return torch.cat(avg_embs), torch.cat(mean_embs), torch.cat(last_embs)
