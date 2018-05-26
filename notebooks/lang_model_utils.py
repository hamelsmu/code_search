from pathlib import Path
import logging
from typing import List, Any
from tqdm import tqdm_notebook
from keras.preprocessing.sequence import pad_sequences
import torch
import spacy
from fastai.text import *
EN = spacy.load('en')


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
        self.bos_token = bos_token

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

    def transform_flattened(self, data: List[str], dedup:bool = True) -> List[int]:
        """Tokenizes, indexes and flattens list of strings for fastai language model."""
        n = len(data)
        logging.warning(f'Transforming {n:,} rows.')
        if dedup:
            data = list(set(data))
            n2 = len(data)
            logging.warning(f'Removed {n-n2:,} duplicate rows.')

        tok_trn = list_flatten([(self.bos_token + ' ' + x).split() for x in data])
        return np.array([self.stoi[s] for s in tok_trn])

    def fit_transform_flattened(self, data: List[str]) -> List[int]:
        "Applies `fit` then `transform_flattened` methods sequentially."
        self.fit(data)
        return self.transform_flattened(data)

    def transform(self,
                 data: List[str],
                 padding: bool = True,
                 max_seq_len: int = 60) -> List[List[int]]:
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
        idx_docs = [[self.stoi[self.bos_token]] + [self.stoi[word] for word in sent.split()[:max_seq_len]] for sent in data]
        # default keras pad_sequences pre-pads to max length with zero
        if padding:
            # because padding currently wrecks hidden state of lang model, so
            # by putting padding after (post) sequence we can just ignore the padding hidden states.
            return pad_sequences(idx_docs, padding='post')
        elif not padding:
            return idx_docs

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
                     lr: float,
                     n_cycle: int = 2,
                     cycle_len: int =3,
                     cycle_mult : int =1,
                     em_sz: int = 400,
                     nh: int = 400,
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

    # borrowed these parameters from fastai
    learner.fit(lr,
                n_cycle=n_cycle,
                wds=wd,
                cycle_len=cycle_len,
                use_clr=(32, 10),
                cycle_mult=cycle_mult,
                best_save_name='langmodel_best')

    # eval sets model to inference mode (turns off dropout, etc.)
    model = learner.model.eval()
    # defensively calling reset to clear model state (b/c its a stateful model)
    model.reset()

    state_dict_dest = mpath/'models/langmodel_best.h5'
    logging.warning(f'State dict for the best model saved here:\n{str(state_dict_dest)}')
    return learner, model


def list2arr(l):
    "Convert list into pytorch Variable."
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
    n_rows = len(list_list_int)
    n_dim = lm_model[0].nhid
    avgarr = np.empty((n_rows, n_dim))
    maxarr = np.empty((n_rows, n_dim))
    lastarr = np.empty((n_rows, n_dim))

    for i in tqdm_notebook(range(len(list_list_int))):
        avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])
        avgarr[i,:] = avg_.data.numpy()
        maxarr[i,:] = max_.data.numpy()
        lastarr[i,:] = last_.data.numpy()

    return avgarr, maxarr, lastarr


def tokenize_docstring(text):
    "Apply tokenization using spacy to docstrings."
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]


class Query2Emb:
    "Assists in turning natural language phrases into sentence embeddings from a language model."
    def __init__(self, lang_model, vocab):
        self.lang_model = lang_model
        self.lang_model.eval()
        self.lang_model.reset()
        self.vocab = vocab
        self.stoi = vocab.stoi
        self.ndim = self._str2emb('This is test to get the dimensionality.').shape[-1]

    def _str2arr(self, str_inp):
        raw_str = ' '.join(tokenize_docstring(str_inp))
        raw_arr = self.vocab.transform([raw_str])[0]
        arr = np.expand_dims(np.array(raw_arr), -1)
        return V(T(arr))

    def _str2emb(self, str_inp):
        v_arr = self._str2arr(str_inp).cpu()
        self.lang_model.reset()
        hidden_states = self.lang_model(v_arr)[-1][-1]
        return hidden_states

    def emb_mean(self, str_inp):
        return self._str2emb(str_inp).mean(0).data.numpy()

    def emb_max(self, str_inp):
        return self._str2emb(str_inp).max(0)[0].data.numpy()

    def emb_last(self, str_inp):
        return self._str2emb(str_inp)[-1].data.numpy()

    def emb_cat(self, str_inp):
        return np.concatenate([self.emb_mean(str_inp), self.emb_max(str_inp), self.emb_last(str_inp)], axis=1)
