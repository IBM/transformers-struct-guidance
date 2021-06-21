from ipdb import set_trace
import os
import sys
import re
import time
import argparse
import random
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
import torch
from factored_emb import *
from tqdm import tqdm
import functools
print = functools.partial(print, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, help='Path to training data.')
parser.add_argument('--dev_data', type=str, help='Path to validation data.')
parser.add_argument('--test_data', type=str, help='Path to test data.')
parser.add_argument('--fpath', type=str, help='File path for estimating surprisals.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.5, help='Hyerparameter in (0, 1) for weighting the structure prediction loss against the word prediction loss. Default is 0.5.')
parser.add_argument('--scaffold_type', type=str, help='Type of scaffold. (next, past)')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--report', type=int, default=1000, help='Frequency of evaluating validation data')
parser.add_argument('--valid_every', type=int, default=10000, help='Frequency of validating and saving model parameters after number of training batches')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--do_train', action='store_true', help='Whether to train the model')
parser.add_argument('--do_test', action='store_true', help='Whether to test the model')
parser.add_argument('--do_eval', action='store_true', help='Whether to use the model for surprisal estimation.')
parser.add_argument('--model_prefix', type=str, default=None, help='Name prefix of the model to be trained and saved')
parser.add_argument('--model', type=str, default=None, help='Path to the trained model checkpoint. Will use the pretrained model if path not specified.')
parser.add_argument('--freeze_embedding', action='store_true', help="Freeze the weights of the input embedding layer.")
parser.add_argument('--batch_size', type=int, default=5, help="Size of a training batch.")
parser.add_argument('--frequent_validation', action='store_true', help="Add frequent validations across epochs")
parser.add_argument('--random_init', action='store_true', help="Randomly initialize model parameters.")
parser.add_argument('--pretokenized', action='store_true', help="Whether input sentences for evaluating surprisals are pertokenized or not.")

args = parser.parse_args()

log_softmax = torch.nn.LogSoftmax(-1)
cross_entropy_loss = torch.nn.CrossEntropyLoss()


def is_nonterminal(token):
    """
    Check if a token is a nonterminal action or word piece token.
    """
    if (token.startswith('NT(') and token.endswith(')')) or token == 'REDUCE()':
        return True
    else:
        return False

def load_sents(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    return lines


def load_data(path, tokenizer):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    data = []

    for line in lines:
        tokens = line.split()
        action_ngrams = []
        words = []
        action_seq = []
        for token in tokens:
            if is_nonterminal(token):
                action_seq.append(token)
            else:
                if action_seq != []:
                    action_ngrams.append('_'.join(action_seq))
                    action_seq = []
                else:
                    action_ngrams.append('_')
                words.append(token)

        action_ngrams.append('_'.join(action_seq)) # add the action ngram that comes after the last word

        sent = ' '.join(words)
        word_pieces = tokenizer.tokenize(sent)

        combined = ''
        n_piece = 0
        word_index = 0
        action_ngram_seq = []

        for piece in word_pieces:
            if piece.startswith(w_boundary_char):
                combined += piece[1:]
            else:
                combined += piece
            n_piece += 1
            if combined == words[word_index]:
                action_ngram_seq += [action_ngrams[word_index]] + ['_' for _ in range(n_piece-1)]
                combined = ''
                n_piece = 0
                word_index += 1
        assert combined == ''
        assert word_index == len(words)

        action_ngram_seq.append(action_ngrams[-1])

        assert len(word_pieces) == (len(action_ngram_seq) - 1)

        data.append([word_pieces, action_ngram_seq]) 
        #set_trace(context=10)
    # for line in tqdm(lines):
    #     tokens = line.split()
    #     action_ngrams = []
    #     words = []
    #     action_seq = []
    #     for token in tokens:
    #         if is_nonterminal(token):
    #             action_seq.append(token)
    #         else:
    #             word_pieces = tokenizer.tokenize(token, add_prefix_space=True)
    #             if action_seq != []:
    #                 action_ngrams += ['_'.join(action_seq)] + ['_' for _ in range(len(word_pieces)-1)]
    #                 action_seq = []
    #             else:
    #                 action_ngrams += ['_' for _ in range(len(word_pieces))]
    #             words += word_pieces
    #     assert action_seq != []
    #     assert len(words) == len(action_ngrams)

        # action_ngrams.append('_'.join(action_seq))
        # words.append('|<end-of-text>|')

        # for w, a in zip(words, action_ngrams):
        #     print('{}\t{}'.format(w, a))

        # data.append([words, action_ngrams])
    return data


def load_sent_parse_pair_data(path):
    '''
    Load data in the format of paired sentence and action ngrams. 
    The sentence and action ngram sequence are separated by "\t".
    e.g.
    "The entire market is at its lowest price-earnings multiple since the dawn of time . <eos>\t
    NT(S)_NT(NP)  _  _  REDUCE()_NT(VP) NT(PP) NT(NP)_NT(NP) _ _ _ REDUCE()_NT(PP) NT(NP)_NT(NP) _ 
    REDUCE()_NT(PP) NT(NP) REDUCE()_REDUCE()_REDUCE()_REDUCE()_REDUCE()_REDUCE()_REDUCE() REDUCE()"
    '''
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    lines = [[line[0].split(), line[1].split()] for line in lines]
    return lines


def get_batches(lines, batch_size):
    if len(lines) % batch_size == 0:
        num_batches = len(lines) // batch_size
    else:
        num_batches = len(lines) // batch_size + 1
    batches = []
    for i in range(num_batches):
        start_index = i*batch_size
        end_index = (i+1)*batch_size
        batch = lines[start_index:end_index]
        batches.append(batch)
    return batches


def get_surprisals(model, tokens, add_bos_token=True):
    surprisals = []
    for i in range(len(tokens)):
        token_id = tokenizer.convert_tokens_to_ids(tokens[i])
        if add_bos_token:
            # add BOS token
            prefix_tokens = [tokenizer.bos_token] + tokens[:i]
        else:
            if i == 0:
                surprisals.append(0.0)
                continue
            else:
                prefix_tokens = tokens[:i]
        ids = tokenizer.convert_tokens_to_ids(prefix_tokens)
        input_ids = torch.tensor([ids]).to(device)
        output = model(input_ids)
        logits = output[0]
        next_token_logits = logits[:, -1, :].squeeze()
        log_probs = log_softmax(next_token_logits)
        surprisal = -log_probs[token_id]/np.log(2)
        surprisals.append(surprisal)
    return surprisals


def get_word_ppl(model, sents, add_bos_token=True):
    nll_total = 0
    word_count = 0
    for sent in sents:
        words = sent.split()
        if add_bos_token:
            tokens = [tokenizer.bos_token] + tokenizer.tokenize(sent)
        else:
            tokens = tokenizer.tokenize(sent)
            if len(tokens) <= 1:
                continue
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(device) # batch size = 1

        loss = model(input_ids, labels=input_ids)[0].item()
        nll_total += loss*(len(tokens)-1)
        word_count += len(words)

    nll_avg = nll_total/word_count
    return np.exp(nll_avg)


def get_validation_loss(model, dev_lines, scaffold_type, add_bos_token=True, word_prediction_loss_only=False):
    if word_prediction_loss_only:
        # Only evaluate word prediction loss
        loss_sum = 0
        token_count = 0

        for line in dev_lines:
            if add_bos_token:
                tokens = [tokenizer.bos_token] + line[0]
            else:
                tokens = line[0]
                if len(tokens) <= 1:
                    continue

            ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device) # batch size = 1

            loss = model(input_ids, labels=input_ids)[0].item()
            loss_sum += loss*(len(tokens)-1)
            token_count += len(tokens) - 1

        loss_avg = loss_sum/token_count

    else:
        word_loss_sum = 0
        action_loss_sum = 0
        word_token_count = 0
        action_token_count = 0

        for line_batch in get_batches(dev_lines, args.batch_size):
            tokens_batch = [[tokenizer.bos_token] + line[0] for line in line_batch]
            n_word_token = np.sum([len(item)-1 for item in tokens_batch])
            n_action_token = n_word_token + len(line_batch)

            # batch_max_len = np.max([len(tokens) for tokens in tokens_batch])
            # tokens_padded_batch = [tokens + [tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

            # attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
            # for b_idx, tokens in enumerate(tokens_batch):
            #     attention_mask[b_idx, :, :, len(tokens):] = 0

            # ids_batch = [tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
            # input_ids = torch.tensor(ids_batch).to(device)
            # action_ids_batch = torch.tensor([[action_vocab.token2index[a_ngram] for a_ngram in line[1]] + [-100 for _ in range(batch_max_len -1 - len(line[1]))] for line in line_batch]).to(device)

            # output  = model(input_ids, labels=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # word_prediction_loss = output[0].item()
            # action_prediction_logits = model.action_decoder(output[-1][-1][:, :-1, :])
            # action_prediction_logits = action_prediction_logits.view(len(tokens_batch)*(batch_max_len-1), -1)
            # action_prediction_loss = cross_entropy_loss(action_prediction_logits, action_ids_batch.view(len(tokens_batch)*(batch_max_len-1), -1).squeeze())

            # loss = (1 - alpha)*word_prediction_loss + alpha*action_prediction_loss

            word_prediction_loss, action_prediction_loss = get_loss(model, line_batch, scaffold_type)

            word_loss_sum += word_prediction_loss.item()*n_word_token 
            action_loss_sum += action_prediction_loss.item()*n_action_token

            word_token_count += n_word_token
            action_token_count += n_action_token

        loss_avg = (1 - alpha)*(word_loss_sum/word_token_count) + alpha*(action_loss_sum/action_token_count)

    return loss_avg


def get_loss(model, line_batch, scaffold_type):
    tokens_batch = [[tokenizer.bos_token] + line[0] for line in line_batch]

    batch_max_len = np.max([len(tokens) for tokens in tokens_batch])
    tokens_padded_batch = [tokens + [tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

    attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
    for b_idx, tokens in enumerate(tokens_batch):
        attention_mask[b_idx, :, :, len(tokens):] = 0

    ids_batch = [tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
    input_ids = torch.tensor(ids_batch).to(device)

    # set_trace(context=10)
    output  = model(input_ids, labels=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    word_prediction_loss = output[0]

    # set_trace(context=10)

    if scaffold_type == 'next':
        action_ids_batch = torch.tensor([[action_vocab.token2index[a_ngram] for a_ngram in line[1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in line_batch]).to(device)
    elif scaffold_type == 'past':
        action_ids_batch = torch.tensor([[action_vocab.pad_index] + [action_vocab.token2index[a_ngram] for a_ngram in line[1][:-1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in line_batch]).to(device)
    else:
        raise NotImplementedError

    action_prediction_logits = model.action_decoder(output[-1][-1][:, :, :])
    action_prediction_logits = action_prediction_logits.view(len(tokens_batch)*batch_max_len, -1)
    action_prediction_loss = cross_entropy_loss(action_prediction_logits, action_ids_batch.view(len(tokens_batch)*batch_max_len, -1).squeeze())

    # loss = (1 - alpha)*word_prediction_loss + alpha*action_prediction_loss

    return word_prediction_loss, action_prediction_loss

    

# load action ngram list and initialize embeddings
with open('bllip-lg_action_ngram_list.txt') as f:
    lines = f.readlines()
symbols = ['<pad>', '_'] + [line.strip().split()[0] for line in lines]
action_vocab = Vocabulary(symbols, 0)

#action_factored_embeddings = FactoredEmbeddings(action_vocab,768, 'base')



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print('Random seed: {}'.format(args.seed), file=sys.stderr)

# Load pretrained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if args.random_init:
    print('Initialize with random weights', file=sys.stderr)
    config = GPT2Config(len(tokenizer))
    model = GPT2LMHeadModel(config).to(device)
else:
    print('Initialize with pretrained weights', file=sys.stderr)
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='pretrained/gpt2').to(device)

# add additional linear layer for predicting action ngram from hidden states
model.action_decoder = torch.nn.Linear(768, len(action_vocab.symbols)).to(device)

w_boundary_char = b'\xc4\xa0'.decode()


if args.model is not None:
    print('Loading parameters from {}'.format(args.model), file=sys.stderr)
    model.load_state_dict(torch.load(args.model))

optimizer = AdamW(model.parameters(), lr=args.lr)

alpha = args.alpha
print('Interpolation weight of structure prediction loss {}'.format(alpha), file=sys.stderr)

SCAFFOLD_TYPE = args.scaffold_type
print('Scaffold type: {}'.format(SCAFFOLD_TYPE), file=sys.stderr)


# Train
if args.do_train:
    # print out training settings
    if args.freeze_embedding:
        print('Freeze embedding layer.', file=sys.stderr)
        model.transformer.wte.weight.requires_grad = False
        model.transformer.wpe.weight.requires_grad = False

    print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
    print('Learning rate: {}'.format(args.lr), file=sys.stderr)
    
    if args.model_prefix is not None:
        model_path = args.model_prefix
    else:
        model_path = "lm_pid{}.params".format(os.getpid())
    print('Model path: {}'.format(model_path), file=sys.stderr)

    train_data_path = args.train_data
    dev_data_path = args.dev_data

    print("Loading train data from {}".format(train_data_path), file=sys.stderr)
    print("Loading dev data from {}".format(dev_data_path), file=sys.stderr)

    train_lines = load_data(train_data_path, tokenizer)
    dev_lines = load_data(dev_data_path, tokenizer)

    n_epochs = args.epochs
    freq_report = args.report
    freq_sample = args.report*10
    freq_valid = args.valid_every

    if args.model is not None:
        model.eval()
        with torch.no_grad():
            valid_loss = get_validation_loss(model, dev_lines, scaffold_type=SCAFFOLD_TYPE)
        print('validation loss: {}'.format(valid_loss))
        best_valid_loss = valid_loss
        model.train()
    else:
        best_valid_loss = np.inf

    for epoch in range(n_epochs):
        random.shuffle(train_lines)
        loss_sum = 0 # cumulative loss of training examples; reset to 0 after every training status report
        line_count = 0 # count of training examples; reset to 0 after every training status report
        count = 0  # cumulative count of training examples
        batch_count = 0 # cumulative count of training batches
        
        for line_batch in get_batches(train_lines, args.batch_size):
            optimizer.zero_grad()

            # set_trace(context=10)

            # tokens_batch = [[tokenizer.bos_token] + line[0] for line in line_batch]

            # batch_max_len = np.max([len(tokens) for tokens in tokens_batch])
            # tokens_padded_batch = [tokens + [tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

            # attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
            # for b_idx, tokens in enumerate(tokens_batch):
            #     attention_mask[b_idx, :, :, len(tokens):] = 0

            # ids_batch = [tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
            # input_ids = torch.tensor(ids_batch).to(device)
            # action_ids_batch = torch.tensor([[action_vocab.token2index[a_ngram] for a_ngram in line[1]] + [-100 for _ in range(batch_max_len -1 - len(line[1]))] for line in line_batch]).to(device)

            # # set_trace(context=10)

            # output  = model(input_ids, labels=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # word_prediction_loss = output[0]

            # # set_trace(context=10)

            # action_prediction_logits = model.action_decoder(output[-1][-1][:, :-1, :])
            # action_prediction_logits = action_prediction_logits.view(len(tokens_batch)*(batch_max_len-1), -1)
            # action_prediction_loss = cross_entropy_loss(action_prediction_logits, action_ids_batch.view(len(tokens_batch)*(batch_max_len-1), -1).squeeze())

            # loss = (1 - alpha)*word_prediction_loss + alpha*action_prediction_loss


            word_prediction_loss, action_prediction_loss = get_loss(model, line_batch, SCAFFOLD_TYPE)

            loss = (1 - alpha)*word_prediction_loss + alpha*action_prediction_loss

            loss.backward()
            optimizer.step()

            # set_trace(context=10)

            loss_sum += loss.item()*len(line_batch)
            count += len(line_batch)
            line_count += len(line_batch)
            batch_count += 1

            if batch_count % freq_report == 0:
                print('Epoch {:.3f} loss: {}'.format(epoch + count/len(train_lines), loss_sum/line_count))
                loss_sum = 0
                line_count = 0

            if batch_count % freq_sample == 0:
                model.eval()
                generated = model.generate(max_length=50, pad_token_id=50256, do_sample=True)
                print(tokenizer.decode(generated[0], skip_special_tokens=True))
                model.train()

            if args.frequent_validation and batch_count % freq_valid == 0:
                model.eval()


                with torch.no_grad():
                    valid_word_prediction_loss = get_validation_loss(model, dev_lines, scaffold_type=SCAFFOLD_TYPE, word_prediction_loss_only=True)
                print('validation next-word prediction loss: {}'.format(valid_word_prediction_loss))
                with torch.no_grad():
                    valid_loss = get_validation_loss(model, dev_lines, scaffold_type=SCAFFOLD_TYPE)
                print('validation loss: {}'.format(valid_loss))

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    # print('new best...')
                    print('new best...writing model to {}'.format(model_path))
                    torch.save(model.state_dict(), model_path)

                model.train()

        model.eval()
        with torch.no_grad():
            valid_loss = get_validation_loss(model, dev_lines, scaffold_type=SCAFFOLD_TYPE)
        print('Epoch {:.3f} loss: {} validation loss: {}'.format(epoch + count/len(train_lines), loss_sum/line_count, valid_loss))
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('new best...writing model to {}'.format(model_path))
            torch.save(model.state_dict(), model_path)

        model.train()
        

if args.do_test:
    model.eval()
    if args.test_data is None:
        raise ValueError('Test data not specified')
    test_data_path = args.test_data
    test_lines = load_data(test_data_path, tokenizer)
    with torch.no_grad():
        test_loss = get_validation_loss(model, test_lines, SCAFFOLD_TYPE, word_prediction_loss_only=True)
    print('Test loss: {}'.format(test_loss))

    if args.test_data is not None:
        test_sents = load_sents(args.test_data)
        with torch.no_grad():
            ppl = get_word_ppl(model, test_sents)
        print('PPL: {}'.format(ppl))



# estimate token surprisal values for unparsed sentences
if args.do_eval:
    model.eval()

    if args.fpath is not None:
        sents = load_sents(args.fpath)
    else:
        sents = ["The dogs under the tree are barking.", "The dogs under the tree is barking.",
                "The keys to the cabinet are on the table.", "The keys to the cabinet is on the table.",
                "No author that liked the senator has ever been popular.", "The author that liked no senator has ever been popular.",]

    print('sentence_id\ttoken_id\ttoken\tsurprisal')

    for i, sent in enumerate(sents):
        if args.pretokenized:
            words = sent.strip().split()
            stimulus = sent.strip()
        else:
            words = nltk.word_tokenize(sent.strip())
            stimulus = ' '.join(words)

        tokens = tokenizer.tokenize(stimulus)
        with torch.no_grad():
            surprisals = get_surprisals(model, tokens, add_bos_token=True)

        index = 0
        for j, word in enumerate(words):
            w_str = ''
            w_surprisal = 0
            while index < len(tokens) and w_str != word:
                token_str = tokens[index]
                if token_str.startswith(w_boundary_char):
                    w_str += token_str[1:]
                else:
                    w_str += token_str
                w_surprisal += surprisals[index]

                index += 1
        
            print('{}\t{}\t{}\t{}'.format(i+1, j+1, word, w_surprisal))

