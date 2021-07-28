from ipdb import set_trace
import os
import sys
import argparse
import random
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
import torch
import functools
import utils
print = functools.partial(print, flush=True)


class Vocabulary(object):
    def __init__(self, symbols, pad_index):
        self.symbols = symbols
        self.pad_index = pad_index
        self.token2index = {}
        for i, symbol in enumerate(symbols):
            self.token2index[symbol] = i
        return

    def pad(self):
        return self.pad_index

    def convert_tokens_to_ids(self, token_input):
        if isinstance(token_input, str):
            return self.token2index[token_input]
        elif isinstance(token_input, list):
            return [self.token2index[token] for token in token_input]
        else:
            raise NotImplementedError

    def convert_ids_to_tokens(self, id_input):
        if isinstance(id_input, int):
            return self.symbols[id_input]
        elif isinstance(id_input, list):
            return [self.symbols[idx] for idx in id_input]
        else:
            raise NotImplementedError


class ScLM:
    def __init__(self, is_random_init, action_ngram_list, device='cuda', model_name='gpt2', cache_dir='pretrained/gpt2'):
        # Load pretrained tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(config).to(device)
        else:
            print('Initialize with pretrained weights', file=sys.stderr)
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir).to(device)

        self.action_vocab = Vocabulary(action_ngram_list, 0)

        self.w_boundary_char = b'\xc4\xa0'.decode()
        self.model.action_decoder = torch.nn.Linear(768, len(self.action_vocab.symbols)).to(device)


    def get_batch_loss(self, line_batch, scaffold_type, add_eos_token=False):
        """
        Assume each line of the batch input contains tokenized sentence paired with action ngram sequence.
        """
        if add_eos_token:
            tokens_batch = [line[0] + [self.tokenizer.bos_token] for line in line_batch]
        else:
            tokens_batch = [line[0] for line in line_batch]

        batch_max_len = np.max([len(tokens) for tokens in tokens_batch])
        tokens_padded_batch = [tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

        # Mask padded tokens
        attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
        for b_idx, tokens in enumerate(tokens_batch):
            attention_mask[b_idx, :, :, len(tokens):] = 0

        input_ids_padded_batch = [self.tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
        input_ids = torch.tensor(input_ids_padded_batch).to(device)

        label_ids_padded_batch = [self.tokenizer.convert_tokens_to_ids(tokens) + [-100 for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]
        label_ids = torch.tensor(label_ids_padded_batch).to(device)

        output  = self.model(input_ids, labels=label_ids, attention_mask=attention_mask, output_hidden_states=True)

        word_prediction_loss = output[0]

        if scaffold_type == 'next':
            action_ids_batch = torch.tensor([[self.action_vocab.token2index[a_ngram] for a_ngram in line[1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in line_batch]).to(device)
        elif scaffold_type == 'past':
            action_ids_batch = torch.tensor([[self.action_vocab.pad_index] + [self.action_vocab.token2index[a_ngram] for a_ngram in line[1][:-1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in line_batch]).to(device)
        else:
            raise NotImplementedError

        action_prediction_logits = self.model.action_decoder(output[-1][-1][:, :, :])
        action_prediction_logits = action_prediction_logits.view(len(tokens_batch)*batch_max_len, -1)
        action_prediction_loss = cross_entropy_loss(action_prediction_logits, action_ids_batch.view(len(tokens_batch)*batch_max_len, -1).squeeze())

        return word_prediction_loss, action_prediction_loss


    def get_validation_loss(self, dev_lines, scaffold_type, word_prediction_loss_only=False):
        if word_prediction_loss_only:
            # Only evaluate word prediction loss
            loss_sum = 0
            token_count = 0

            for line in dev_lines:
                tokens = line[0]
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor([ids]).to(device) # batch size = 1

                loss = self.model(input_ids, labels=input_ids)[0].item()
                loss_sum += loss*(len(tokens)-1)
                token_count += len(tokens) - 1

            loss_avg = loss_sum/token_count

            return loss_avg

        else:
            word_loss_sum = 0
            action_loss_sum = 0
            word_token_count = 0
            action_token_count = 0

            for line_batch in get_batches(dev_lines, args.batch_size):
                n_word_token = np.sum([len(word_tokens) - 1 for word_tokens, _ in line_batch])
                n_action_token = n_word_token + len(line_batch)

                word_prediction_loss, action_prediction_loss = self.get_batch_loss(line_batch, scaffold_type, add_eos_token=False)

                word_loss_sum += word_prediction_loss.item()*n_word_token 
                action_loss_sum += action_prediction_loss.item()*n_action_token

                word_token_count += n_word_token
                action_token_count += n_action_token

            word_loss_avg = word_loss_sum/word_token_count
            action_loss_avg = action_loss_sum/action_token_count
            loss_avg = (1 - ALPHA)*word_loss_avg + ALPHA*action_loss_avg

            return loss_avg, word_loss_avg, action_loss_avg


    def get_surprisals(self, tokens, add_bos_token=True):
        surprisals = []
        for i in range(len(tokens)):
            token_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            if add_bos_token:
                # add BOS token
                prefix_tokens = [self.tokenizer.bos_token] + tokens[:i]
            else:
                if i == 0:
                    surprisals.append(0.0)
                    continue
                else:
                    prefix_tokens = tokens[:i]

            ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
            input_ids = torch.tensor([ids]).to(device)
            output = self.model(input_ids)
            logits = output[0]
            next_token_logits = logits[:, -1, :].squeeze()
            log_probs = log_softmax(next_token_logits)
            surprisal = -log_probs[token_id]/np.log(2)
            surprisals.append(surprisal)
        return surprisals


    def get_word_ppl(self, sents, add_bos_token=True):
        nll_total = 0
        word_count = 0

        total_token_count = 0

        for sent in sents:
            words = sent.split()
            if add_bos_token:
                tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(sent)
            else:
                tokens = self.tokenizer.tokenize(sent)
                if len(tokens) <= 1:
                    continue

            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device) # batch size = 1

            loss = self.model(input_ids, labels=input_ids)[0].item() # batch size = 1
            nll_total += loss*(len(tokens)-1)
            word_count += len(words)

            total_token_count += len(tokens)-1

        #print(nll_total, word_count, total_token_count)
        nll_avg = nll_total/word_count
        return np.exp(nll_avg)


    def generate(self, prompt, max_len=50, top_k=50, top_p=0.92, temperature=1, n_sample=1, device='cuda'):
        """
        Sample from the model.
        """
        tokens = self.tokenizer.tokenize(prompt)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(device)
        output_ids_batch = self.model.generate(input_ids, do_sample=True, max_length=max_len, pad_token_id=50256,
                                                top_k=top_k, top_p=top_p, temperature=temperature, num_return_sequences=n_sample)
        samples = [self.tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in output_ids_batch]
        return samples


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


def load_data(path, tokenizer, BOS_token=None):

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

        if BOS_token is None:
            data.append([word_pieces, action_ngram_seq]) 
        else:
            data.append([[BOS_token] + word_pieces, action_ngram_seq]) 

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='Path to training data.')
    parser.add_argument('--dev_data', type=str, help='Path to validation data.')
    parser.add_argument('--test_data', type=str, help='Path to test data.')
    parser.add_argument('--fpath', type=str, help='File path for estimating surprisals.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='Hyerparameter in (0, 1) for weighting the structure prediction loss against the word prediction loss. Default is 0.5.')
    parser.add_argument('--scaffold_type', type=str, help='Type of scaffold. (next, past)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--report', type=int, default=1000, help='Frequency of report training status after number of training batches')
    parser.add_argument('--valid_every', type=int, default=None, help='Frequency of validating and saving model parameters after number of training batches')
    parser.add_argument('--sample_every', type=int, default=10000, help='Frequency of generating samples from the model during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--do_train', action='store_true', help='Whether to train the model')
    parser.add_argument('--do_test', action='store_true', help='Whether to test the model')
    parser.add_argument('--do_eval', action='store_true', help='Whether to use the model for surprisal estimation.')
    parser.add_argument('--model_path', type=str, default=None, help='Path of the model to be trained and saved')
    parser.add_argument('--restore_from', type=str, default=None, help='Path to the trained model checkpoint. Will use the pretrained model if path not specified.')
    parser.add_argument('--batch_size', type=int, default=5, help="Size of a training batch.")
    parser.add_argument('--early_stopping_threshold', type=int, default=2, help='Threshold for early stopping.')
    parser.add_argument('--random_init', action='store_true', help="Randomly initialize model parameters.")
    parser.add_argument('--pretokenized', action='store_true', help="Whether input sentences for evaluating surprisals are pertokenized or not.")

    args = parser.parse_args()

    log_softmax = torch.nn.LogSoftmax(-1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

    # load action ngram list and initialize embeddings
    with open('bllip-lg_action_ngram_list.txt') as f:
        lines = f.readlines()
    symbols = ['<pad>', '_'] + [line.strip().split()[0] for line in lines]

    # Initialize the model
    sclm = ScLM(is_random_init=args.random_init, action_ngram_list=symbols, device=device, model_name='gpt2')
    w_boundary_char = sclm.w_boundary_char

    # Load model checkpoint
    if args.restore_from is not None:
        print('Load parameters from {}'.format(args.restore_from), file=sys.stderr)
        checkpoint = torch.load(args.restore_from)
        sclm.model.load_state_dict(checkpoint['model_state_dict'])

    SCAFFOLD_TYPE = args.scaffold_type
    print('Scaffold type: {}'.format(SCAFFOLD_TYPE), file=sys.stderr)
    ALPHA = args.alpha
    print('Interpolation weight of structure prediction loss {}'.format(ALPHA), file=sys.stderr)


    # Train
    if args.do_train:
        # Path to save the newly trained model
        MODEL_PATH = args.model_path if args.model_path is not None else "sclm-{}_pid{}.params".format(SCAFFOLD_TYPE, os.getpid())

        # print out training settings
        print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
        print('Learning rate: {}'.format(args.lr), file=sys.stderr)
        print('Model path: {}'.format(MODEL_PATH), file=sys.stderr)

        optimizer = AdamW(sclm.model.parameters(), lr=args.lr)

        # Load train and dev data
        train_data_path = args.train_data
        dev_data_path = args.dev_data
        print("Loading train data from {}".format(train_data_path), file=sys.stderr)
        train_lines = load_data(train_data_path, sclm.tokenizer, BOS_token=sclm.tokenizer.bos_token)
        print("Loading dev data from {}".format(dev_data_path), file=sys.stderr)
        dev_lines = load_data(dev_data_path, sclm.tokenizer, BOS_token=sclm.tokenizer.bos_token)


        if args.restore_from is not None:
            sclm.model.eval()
            with torch.no_grad():
                validation_loss, _, _ = sclm.get_validation_loss(dev_lines, scaffold_type=SCAFFOLD_TYPE)
            best_validation_loss = validation_loss
            sclm.model.train()
            print('resume training; validation loss: {}'.format(best_validation_loss))
        else:
            best_validation_loss = np.inf

        n_epochs = args.epochs
        starting_epoch = checkpoint['epoch'] + 1 if (args.restore_from is not None) else 0
        no_improvement_count = checkpoint['no_improvement_count'] if (args.restore_from is not None) else 0
        VALID_EVERY = None if ((args.valid_every is None) or (args.valid_every < 1)) else args.valid_every
        
        early_stopping_counter = utils.EarlyStopping(best_validation_loss=best_validation_loss, no_improvement_count=no_improvement_count, threshold=args.early_stopping_threshold)

        for epoch in range(starting_epoch, n_epochs):
            random.shuffle(train_lines)

            count = 0  # cumulative count of training examples
            batch_count = 0 # cumulative count of training batches
            
            for line_batch in get_batches(train_lines, args.batch_size):
                optimizer.zero_grad()

                word_prediction_loss, action_prediction_loss = sclm.get_batch_loss(line_batch, SCAFFOLD_TYPE, add_eos_token=True)

                loss = (1 - ALPHA)*word_prediction_loss + ALPHA*action_prediction_loss

                loss.backward()
                optimizer.step()

                count += len(line_batch)
                batch_count += 1

                if batch_count > 0 and batch_count % args.report == 0:
                    print('Epoch {:.3f} loss: {}'.format(epoch + count/len(train_lines), loss.item()))

                if batch_count > 0  and batch_count % args.sample_every == 0:
                    sclm.model.eval()
                    with torch.no_grad():
                        samples = sclm.generate(prompt="<|endoftext|>")
                        for sample in samples:
                            print(sample)
                    sclm.model.train()


                if VALID_EVERY is not None:
                    if batch_count > 0  and batch_count % VALID_EVERY == 0:
                        sclm.model.eval()

                        with torch.no_grad():
                            validation_loss, validation_word_prediction_loss, _ = sclm.get_validation_loss(dev_lines, scaffold_type=SCAFFOLD_TYPE)
                        print('Epoch {:.3f} validation loss: {} validation next-word prediction loss: {}'.format(epoch + count/len(train_lines), validation_loss, validation_word_prediction_loss))

                        is_early_stop = early_stopping_counter.check_stopping_criterion(validation_loss)
                        if is_early_stop:
                            print('Validation loss increases for {} epochs in a row.'.format(early_stopping_counter.counter))
                            print('EARLY STOPPING...')
                            sys.exit()

                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            print("new best... saving model to {}".format(MODEL_PATH))
                            torch.save(
                                {'epoch': epoch,
                                'scaffold_type': SCAFFOLD_TYPE,
                                'no_improvement_count': early_stopping_counter.counter,
                                'model_state_dict': sclm.model.state_dict(),
                                'loss': validation_loss}, MODEL_PATH)

                        sclm.model.train()


            sclm.model.eval()

            with torch.no_grad():
                validation_loss, validation_word_prediction_loss, _ = sclm.get_validation_loss(dev_lines, scaffold_type=SCAFFOLD_TYPE)
            print('Epoch {} validation loss: {} validation next-word prediction loss: {}'.format(epoch, validation_loss, validation_word_prediction_loss))

            is_early_stop = early_stopping_counter.check_stopping_criterion(validation_loss)
            if is_early_stop:
                print('Validation loss increases for {} epochs in a row.'.format(early_stopping_counter.counter))
                print('EARLY STOPPING...')
                sys.exit()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                print("new best... saving model to {}".format(MODEL_PATH))
                torch.save(
                    {'epoch': epoch,
                    'scaffold_type': SCAFFOLD_TYPE,
                    'no_improvement_count': early_stopping_counter.counter,
                    'model_state_dict': sclm.model.state_dict(),
                    'loss': validation_loss}, MODEL_PATH)

            sclm.model.train()
            

    if args.do_test:
        sclm.model.eval()
        if args.test_data is None:
            raise ValueError('Test data not specified')

        test_data_path = args.test_data

        test_sents = []
        lines = load_sents(args.test_data)
        for line in lines:
            tokens = line.split()
            words = [token for token in tokens if not is_nonterminal(token)]
            test_sents.append(' '.join(words))

        with torch.no_grad():
            ppl = sclm.get_word_ppl(test_sents)
        print('PPL: {}'.format(ppl))


    # Estimate token surprisal values for unparsed sentences
    if args.do_eval:
        sclm.model.eval()

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

            tokens = sclm.tokenizer.tokenize(stimulus)
            with torch.no_grad():
                surprisals = sclm.get_surprisals(tokens, add_bos_token=True)

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

