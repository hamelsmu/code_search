from fastai.text import *
from pathlib import Path
import logging
BOS = '_xbos_ '

def list_flatten(l):
    "List[List] --> List"
    return [item for sublist in l for item in sublist]

def preprocess_lm_data(read_data_path,
                       save_data_path,
                       train_file='train.docstring',
                       validation_file='valid.docstring',
                       max_vocab=50000,
                       min_freq=15):
    """
    Pre-process data for fast.ai language model.
    
    Parameters
    ==========
    read_data_path : str
        Path where raw data will be read from.
        
    save_data_path : str
        Path where transformed data and dictionaries will be saved.
        The following files will be saved in this path:
        1. itos_dict.pkl  this a dict that maps integer indices to a string.
        2. stoi_dict.pkl  this is a defaultdict which maps strings to integer indices,
                           with default value of zero.
        3. trn_indexed.npy a numpy array that is the indexed version of the training data.
        4. val_indexed.npy a numpy array that is the indexed version of the validation data.
     
     train_file : str
         The name of the training file in the `read_data_path` defaults to 'train.docstring'

     val_file : str
         The name of the validation file in the `read_data_path` defaults to 'valid.docstring'
         
     max_vocab : int
          Maximum sie of vocabulary.
     
     min_freq : int
          Minimum frequency threshold for token to be included in vocabulary.
     
     Returns
     =======
     None
         This function saves 4 files to the specified `save_data_path` but does not return anything.
    """
    read_path = Path(read_data_path)
    with open(read_path/train_file, 'r') as f:
        t_comment = f.readlines()
    logging.warning(f'Training file has {len(t_comment):,} rows')
    
    with open(read_path/validation_file, 'r') as f:
        v_comment = f.readlines()
    logging.warning(f'Validation file has {len(v_comment):,} rows')
    
    # flatten raw data
    tok_trn = list_flatten([(BOS + x).split() for x in t_comment])
    tok_val = list_flatten([(BOS + x).split() for x in v_comment])

    # Build vocabulary dicts: index to string (itos), string to index(stoi)
    freq = Counter(tok_trn) # on training set, then applied to val
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    logging.warning(f'Vocab Size {len(itos):,}')
    
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    
    # turn sequence of strings into sequence of integers
    trn_indexed = np.array([stoi[s] for s in tok_trn])
    val_indexed = np.array([stoi[s] for s in tok_val])
    
    # save all artifacts in `save_data_path`
    save_path = Path(save_data_path)
    save_path.mkdir(exist_ok=True)
    with open(save_path/'itos_dict.pkl', 'wb') as f:
        pickle.dump(itos, f)
    
    with open(save_path/'stoi_dict.pkl', 'wb') as f:
        pickle.dump(itos, f)
        
    np.save(save_path/'trn_indexed.npy', trn_indexed)
    np.save(save_path/'val_indexed.npy', val_indexed)
    
    
def read_preprocessed_files(processed_data_path):
    """
    Read preprocessed files for fast.ai language model.
    
    Parameters
    ==========
    processed_data_path : str
        The path where your processed data is stored.  This
        is a result of running the `preprocess_lm_data` function.
    
    Returns
    ==========
    Tuple(dict, dict, np.array, np.array)
    """
    PATH = Path(processed_data_path)
    
    with open(PATH/'itos_dict.pkl', 'rb') as f:
        itos = pickle.load(f)
        
    with open(PATH/'stoi_dict.pkl', 'rb') as f:
        stoi = pickle.load(f)
        
    
    trn_indexed = np.load(PATH/'trn_indexed.npy')
    val_indexed = np.load(PATH/'val_indexed.npy')
    
    return itos, stoi, trn_indexed, val_indexed


def train_lang_model(model_path, 
                     data_path,
                     n_cycle=2,
                     em_sz=800,
                     nh=1200,
                     nl=3,
                     bptt=20,
                     wd=1e-7,
                     bs=32):
    """
    Train fast.ai language model.
    
    Parameters
    ==========
    model_path : str
        Path where you want to save model artifacts.
    data_path : str
        Path where we can retrieve the data for training the model.
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
    PATH = Path(model_path)
    PATH.mkdir(exist_ok=True)
    
    itos, stoi, trn_indexed, val_indexed = read_preprocessed_files(data_path)
    vs=len(itos)
    
    # create data loaders
    trn_dl = LanguageModelLoader(trn_indexed, bs, bptt)
    val_dl = LanguageModelLoader(val_indexed, bs, bptt)
    
    # create lang model data
    md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)
    
    # build learner
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7
    
    learner = md.get_model(opt_fn, em_sz, nh, nl, 
                           dropouti=drops[0], 
                           dropout=drops[1], 
                           wdrop=drops[2], 
                           dropoute=drops[3], 
                           dropouth=drops[4])
    lrs = 1e-3 / 2
    
    learner.fit(lrs, 2, wds=wd, cycle_len=3, use_clr=(32,10), best_save_name='langmodel_best.torch')
    model = learner.model.eval()
    model.reset()
    return learner, model 

def list2arr(l):
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
    