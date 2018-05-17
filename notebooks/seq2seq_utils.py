from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import logging
import numpy as np
import dill as dpickle
from annoy import AnnoyIndex
from tqdm import tqdm, tqdm_notebook
from random import random
from nltk.translate.bleu_score import corpus_bleu


def build_seq2seq_model(word_emb_dim,
                        hidden_state_dim,
                        encoder_seq_len,
                        num_encoder_tokens,
                        num_decoder_tokens):
    """
    Builds architecture for sequence to sequence model.

    Encoder and Decoder layer consist of one GRU Layer each.  User
    can specify the dimensionality of the word embedding and the hidden state.

    Parameters
    ----------
    word_emb_dim : int
        dimensionality of the word embeddings
    hidden_state_dim : int
        dimensionality of the hidden state in the encoder and decoder.
    encoder_seq_len : int
        the length of the sequences that are input into the encoder.  The
        sequences are expected to all be padded to the same size.
    num_encoder_tokens : int
        the vocabulary size of the corpus relevant to the encoder.
    num_decoder_tokens : int
        the vocabulary size of the corpus relevant to the decoder.

    Returns
    -------
    Keras.models.Model
    """

    #### Encoder Model ####
    encoder_inputs = Input(shape=(encoder_seq_len,), name='Encoder-Input')

    # Word embeding for encoder (ex: Issue Titles, Code)
    x = Embedding(num_encoder_tokens, word_emb_dim, name='Body-Word-Embedding', mask_zero=False)(encoder_inputs)
    x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

    # We do not need the `encoder_output` just the hidden state.
    _, state_h = GRU(hidden_state_dim, return_state=True, name='Encoder-Last-GRU', dropout=.5)(x)

    # Encapsulate the encoder as a separate entity so we can just
    #  encode without decoding if we want to.
    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')

    seq2seq_encoder_out = encoder_model(encoder_inputs)

    #### Decoder Model ####
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

    # Word Embedding For Decoder (ex: Issue Titles, Docstrings)
    dec_emb = Embedding(num_decoder_tokens, word_emb_dim, name='Decoder-Word-Embedding', mask_zero=False)(decoder_inputs)
    dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

    # Set up the decoder, using `decoder_state_input` as initial state.
    decoder_gru = GRU(hidden_state_dim, return_state=True, return_sequences=True, name='Decoder-GRU', dropout=.5)
    decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
    x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

    # Dense layer for prediction
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')
    decoder_outputs = decoder_dense(x)

    #### Seq2Seq Model ####
    seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return seq2seq_Model


def load_text_processor(fname='title_pp.dpkl'):
    """
    Load preprocessors from disk.

    Parameters
    ----------
    fname: str
        file name of ktext.proccessor object

    Returns
    -------
    num_tokens : int
        size of vocabulary loaded into ktext.processor
    pp : ktext.processor
        the processor you are trying to load

    Typical Usage:
    -------------

    num_decoder_tokens, title_pp = load_text_processor(fname='title_pp.dpkl')
    num_encoder_tokens, body_pp = load_text_processor(fname='body_pp.dpkl')

    """
    # Load files from disk
    with open(fname, 'rb') as f:
        pp = dpickle.load(f)

    num_tokens = max(pp.id2token.keys()) + 1
    print(f'Size of vocabulary for {fname}: {num_tokens:,}')
    return num_tokens, pp


def load_decoder_inputs(decoder_np_vecs='train_title_vecs.npy'):
    """
    Load decoder inputs.

    Parameters
    ----------
    decoder_np_vecs : str
        filename of serialized numpy.array of decoder input (issue title)

    Returns
    -------
    decoder_input_data : numpy.array
        The data fed to the decoder as input during training for teacher forcing.
        This is the same as `decoder_np_vecs` except the last position.
    decoder_target_data : numpy.array
        The data that the decoder data is trained to generate (issue title).
        Calculated by sliding `decoder_np_vecs` one position forward.

    """
    vectorized_title = np.load(decoder_np_vecs)
    # For Decoder Input, you don't need the last word as that is only for prediction
    # when we are training using Teacher Forcing.
    decoder_input_data = vectorized_title[:, :-1]

    # Decoder Target Data Is Ahead By 1 Time Step From Decoder Input Data (Teacher Forcing)
    decoder_target_data = vectorized_title[:, 1:]

    print(f'Shape of decoder input: {decoder_input_data.shape}')
    print(f'Shape of decoder target: {decoder_target_data.shape}')
    return decoder_input_data, decoder_target_data


def load_encoder_inputs(encoder_np_vecs='train_body_vecs.npy'):
    """
    Load variables & data that are inputs to encoder.

    Parameters
    ----------
    encoder_np_vecs : str
        filename of serialized numpy.array of encoder input (issue title)

    Returns
    -------
    encoder_input_data : numpy.array
        The issue body
    doc_length : int
        The standard document length of the input for the encoder after padding
        the shape of this array will be (num_examples, doc_length)

    """
    vectorized_body = np.load(encoder_np_vecs)
    # Encoder input is simply the body of the issue text
    encoder_input_data = vectorized_body
    doc_length = encoder_input_data.shape[1]
    print(f'Shape of encoder input: {encoder_input_data.shape}')
    return encoder_input_data, doc_length


def viz_model_architecture(model):
    """Visualize model architecture in Jupyter notebook."""
    display(SVG(model_to_dot(model).create(prog='dot', format='svg')))


def free_gpu_mem():
    """Attempt to free gpu memory."""
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def test_gpu():
    """Run a toy computation task in tensorflow to test GPU."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    hello = tf.constant('Hello, TensorFlow!')
    print(session.run(hello))


def plot_model_training_history(history_object):
    """Plots model train vs. validation loss."""
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def extract_encoder_model(model):
    """
    Extract the encoder from the original Sequence to Sequence Model.

    Returns a keras model object that has one input (body of issue) and one
    output (encoding of issue, which is the last hidden state).

    Input:
    -----
    model: keras model object

    Returns:
    -----
    keras model object

    """
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model


def extract_decoder_model(model):
    """
    Extract the decoder from the original model.

    Inputs:
    ------
    model: keras model object

    Returns:
    -------
    A Keras model object with the following inputs and outputs:

    Inputs of Keras Model That Is Returned:
    1: the embedding index for the last predicted word or the <Start> indicator
    2: the last hidden state, or in the case of the first word the hidden state from the encoder

    Outputs of Keras Model That Is Returned:
    1.  Prediction (class probabilities) for the next word
    2.  The hidden state of the decoder, to be fed back into the decoder at the next time step

    Implementation Notes:
    ----------------------
    Must extract relevant layers and reconstruct part of the computation graph
    to allow for different inputs as we are not going to use teacher forcing at
    inference time.

    """
    # the latent dimension is the dmeinsion of the hidden state passed from the encoder to the decoder.
    latent_dim = model.get_layer('Encoder-Model').output_shape[-1]

    # Reconstruct the input into the decoder
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    # Instead of setting the intial state from the encoder and forgetting about it, during inference
    # we are not doing teacher forcing, so we will have to have a feedback loop from predictions back into
    # the GRU, thus we define this input layer for the state so we can add this capability
    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')

    # we need to reuse the weights that is why we are getting this
    # If you inspect the decoder GRU that we created for training, it will take as input
    # 2 tensors -> (1) is the embedding layer output for the teacher forcing
    #                  (which will now be the last step's prediction, and will be _start_ on the first time step)
    #              (2) is the state, which we will initialize with the encoder on the first time step, but then
    #                   grab the state after the first prediction and feed that back in again.
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input])

    # Reconstruct dense layers
    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
    decoder_model = Model([decoder_inputs, gru_inference_state_input],
                          [dense_out, gru_state_out])
    return decoder_model


class Seq2Seq_Inference(object):
    def __init__(self,
                 encoder_preprocessor,
                 decoder_preprocessor,
                 seq2seq_model):

        self.enc_pp = encoder_preprocessor
        self.dec_pp = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len = self.dec_pp.padding_maxlen
        self.nn = None
        self.rec_df = None

    def predict(self,
                raw_input_text,
                max_len=None):
        """
        Use the seq2seq model to generate a output given the input.

        Inputs
        ------
        raw_input: str
            The body of what is to be summarized or translated.

        max_len: int (optional)
            The maximum length of the output

        """
        if max_len is None:
            max_len = self.default_max_len
        # get the encoder's features for the decoder
        raw_tokenized = self.enc_pp.transform([raw_input_text])
        encoding = self.encoder_model.predict(raw_tokenized)
        # we want to save the encoder's embedding before its updated by decoder
        #   because we can use that as an embedding for other tasks.
        original_encoding = encoding
        state_value = np.array(self.dec_pp.token2id['_start_']).reshape(1, 1)

        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict([state_value, encoding])

            # We are going to ignore indices 0 (padding) and indices 1 (unknown)
            # Argmax will return the integer index corresponding to the
            #  prediction + 2 b/c we chopped off first two
            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            # retrieve word from index prediction
            pred_word_str = self.dec_pp.id2token[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_encoding, ' '.join(decoded_sentence)


    def print_example(self,
                      i,
                      input_text,
                      output_text,
                      url,
                      threshold):
        """
        Prints an example of the model's prediction for manual inspection.
        """
        if i:
            print('\n\n==============================================')
            print(f'============== Example # {i} =================\n')

        if url:
            print(url)

        print(f"Original Input:\n {input_text} \n")

        if output_text:
            print(f"Original Output:\n {output_text}")

        emb, gen_title = self.predict(input_text)
        print(f"\n****** Predicted Output ******:\n {gen_title}")


    def demo_model_predictions(self,
                               n,
                               df,
                               threshold=1,
                               input_col='code',
                               output_col='comment',
                               ref_col='ref'):
        """
        Pick n random Issues and display predictions.

        Input:
        ------
        n : int
            Number of examples to display from
        df : pandas DataFrame
        threshold : float
            distance threshold for recommendation of similar issues.

        Returns:
        --------
        None
            Prints the original issue body and the model's prediction.
        """
        # Extract input and output from DF
        input_text = df[input_col].tolist()
        output_text = df[output_col].tolist()
        url = df[ref_col].tolist()

        demo_list = np.random.randint(low=1, high=len(input_text), size=n)
        for i in demo_list:
            self.print_example(i,
                               input_text=input_text[i],
                               output_text=output_text[i],
                               url=url[i],
                               threshold=threshold)

    def evaluate_model(self, input_strings, output_strings, max_len):
        """
        Method for calculating BLEU Score.

        Parameters
        ----------
        input_strings : List[str]
            These are the issue bodies that we want to summarize
        output_strings : List[str]
            This is the ground truth we are trying to predict --> issue titles

        Returns
        -------
        bleu : float
            The BLEU Score

        """
        self.actual, self.predicted = list(), list()
        assert len(input_strings) == len(output_strings)
        num_examples = len(input_strings)

        logging.warning('Generating predictions.')
        # step over the whole set TODO: parallelize this
        for i in tqdm_notebook(range(num_examples)):
            _, yhat = self.predict(input_strings[i], max_len)

            self.actual.append(self.dec_pp.process_text([output_strings[i]])[0])
            self.predicted.append(self.dec_pp.process_text([yhat])[0])
        # calculate BLEU score
        logging.warning('Calculating BLEU.')
        bleu = corpus_bleu([[a] for a in self.actual], self.predicted)
        return bleu
