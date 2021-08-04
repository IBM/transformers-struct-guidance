import os
import sys
import argparse
import random
import numpy as np
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
import torch
import utils
import functools
print = functools.partial(print, flush=True)


class LM:
    def __init__(self, is_random_init, device='cuda', model_name='gpt2', cache_dir='pretrained/gpt2'):
        # Load pretrained tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(config).to(device)
        else:
            print('Initialize with pretrained weights', file=sys.stderr)
            self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)

        self.w_boundary_char = b'\xc4\xa0'.decode()

    def get_batch_loss(self, data_batch, device='cuda'):
        """
        Assume a data batch as a list of sequences.
        """
        tokens_batch = [self.tokenizer.tokenize(line) for line in data_batch]

        token_count_batch = [len(tokens) for tokens in tokens_batch]
        batch_max_len = np.max(token_count_batch)

        tokens_padded_batch = [tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

        attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
        for b_idx, tokens in enumerate(tokens_batch):
            attention_mask[b_idx, :, :, len(tokens):] = 0

        input_ids_padded_batch = [self.tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
        input_ids = torch.tensor(input_ids_padded_batch).to(device)
        label_ids_padded_batch = [self.tokenizer.convert_tokens_to_ids(tokens) + [-100 for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]
        label_ids = torch.tensor(label_ids_padded_batch).to(device)

        output = self.model(input_ids, labels=label_ids, attention_mask=attention_mask, return_dict=True)

        loss = output.loss
        batch_token_count = np.sum(token_count_batch) - len(tokens_batch) # substract the count since len(tokens)-1 words are counted
        return loss, batch_token_count

    def get_loss(self, data, batch_size, device='cuda'):
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(data, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item()*batch_token_count
            total_token_count += batch_token_count

        return total_loss/total_token_count

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

    def get_word_level_perplexity(self, dev_lines, add_bos_token=True):
        loss_sum = 0
        total_word_count = 0

        for line in dev_lines:
            if add_bos_token:
                tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(line)
            else:
                tokens = self.tokenizer.tokenize(line)
                if len(tokens) <= 1:
                    continue

            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device) # batch size = 1

            loss = self.model(input_ids, labels=input_ids)[0].item()
            loss_sum += loss*(len(tokens)-1)
            total_word_count += len(line.strip().split())
        return np.exp(loss_sum/total_word_count)

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


def load_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
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
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--report', type=int, default=1000, help='Frequency of reporting training status after number of training batches.')
    parser.add_argument('--sample_every', type=int, default=10000, help='Frequency of generating samples from the model during training.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--do_train', action='store_true', help='Whether to train the model.')
    parser.add_argument('--do_test', action='store_true', help='Whether to test the model.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to use the model for surprisal estimation.')
    parser.add_argument('--model_path', type=str, default=None, help='Path of the model to be trained and saved.')
    parser.add_argument('--restore_from', type=str, default=None, help='Path to the trained model checkpoint. Will use the pretrained model if path not specified.')
    parser.add_argument('--batch_size', type=int, default=5, help="Size of a training batch.")
    parser.add_argument('--early_stopping_threshold', type=int, default=2, help='Threshold for early stopping.')
    parser.add_argument('--random_init', action='store_true', help="Randomly initialize model parameters.")
    parser.add_argument('--fpath', type=str, help='Path to text file for estimating surprisals.')
    parser.add_argument('--pretokenized', action='store_true', help="Whether input sentences for evaluating surprisals are pertokenized or not.")

    args = parser.parse_args()

    log_softmax = torch.nn.LogSoftmax(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

    # Initialize language model
    lm = LM(is_random_init=args.random_init, device=device, model_name='gpt2', cache_dir='pretrained/gpt2')

    # Restore from a model checkpoint
    if args.restore_from is not None:
        print('Load parameters from {}'.format(args.restore_from), file=sys.stderr)
        checkpoint = torch.load(args.restore_from)
        # lm.model.load_state_dict(torch.load(args.restore_from))
        lm.model.load_state_dict(checkpoint['model_state_dict'])

    # Train
    if args.do_train:
        # Path to save the newly trained model
        MODEL_PATH = args.model_path if args.model_path is not None else "lm_pid{}.params".format(os.getpid())

        # Set the learning rate of the optimizer
        optimizer = AdamW(lm.model.parameters(), lr=args.lr)

        # Print out training settings
        print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
        print('Learning rate: {}'.format(args.lr), file=sys.stderr)
        print('Model path: {}'.format(MODEL_PATH), file=sys.stderr)

        # Load train and dev data
        train_data_path = args.train_data
        dev_data_path = args.dev_data
        print("Loading train data from {}".format(train_data_path), file=sys.stderr)
        print("Loading dev data from {}".format(dev_data_path), file=sys.stderr)
        train_lines = load_data(train_data_path)
        dev_lines = load_data(dev_data_path)
        train_lines = [lm.tokenizer.bos_token + line + lm.tokenizer.bos_token for line in train_lines]
        dev_lines = [lm.tokenizer.bos_token + line for line in dev_lines]

        if args.restore_from is not None:
            lm.model.eval()
            with torch.no_grad():
                best_validation_loss = lm.get_loss(dev_lines, batch_size=args.batch_size)
            lm.model.train()
            print('resume training; validation loss: {}'.format(best_validation_loss))
        else:
            best_validation_loss = np.Inf

        n_epochs = args.epochs
        starting_epoch = checkpoint['epoch'] + 1 if (args.restore_from is not None) else 0
        no_improvement_count = checkpoint['no_improvement_count'] if (args.restore_from is not None) else 0

        early_stopping_counter = utils.EarlyStopping(best_validation_loss=best_validation_loss, no_improvement_count=no_improvement_count, threshold=args.early_stopping_threshold)

        for epoch in range(starting_epoch, n_epochs):
            random.shuffle(train_lines)

            count = 0  # cumulative count of training examples
            batch_count = 0 # cumulative count of training batches

            for train_data_batch in get_batches(train_lines, args.batch_size):
                optimizer.zero_grad()

                loss, batch_token_count = lm.get_batch_loss(train_data_batch)
                loss.backward()
                optimizer.step()

                batch_count += 1
                count += len(train_data_batch)

                if batch_count > 0 and batch_count % args.report == 0:
                    print('Epoch {:.3f} loss: {}'.format(epoch + count/len(train_lines), loss.item()))

                if batch_count > 0  and batch_count % args.sample_every == 0:
                    lm.model.eval()
                    with torch.no_grad():
                        samples = lm.generate(prompt="<|endoftext|>")
                        for sample in samples:
                            print(sample)
                    lm.model.train()

            lm.model.eval()
            with torch.no_grad():
                validation_loss = lm.get_loss(dev_lines, batch_size=args.batch_size)
            print('Epoch', epoch, 'validation loss:', validation_loss)

            is_early_stop = early_stopping_counter.check_stopping_criterion(validation_loss)
            if is_early_stop:
                print('Validation loss increases for {} epochs in a row.'.format(early_stopping_counter.counter))
                print('EARLY STOPPING...')
                break

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                print("new best... saving model to {}".format(MODEL_PATH))
                torch.save(
                    {'epoch': epoch,
                    'no_improvement_count': early_stopping_counter.counter,
                    'model_state_dict': lm.model.state_dict(),
                    'loss': validation_loss}, MODEL_PATH)

            lm.model.train()

    # Test
    if args.do_test:
        lm.model.eval()
        if args.test_data is None:
            raise ValueError('Test data not specified')

        test_lines = load_data(args.test_data)
        test_lines = [lm.tokenizer.bos_token + '' + line for line in test_lines]

        with torch.no_grad():
            validation_loss = lm.get_loss(test_lines, batch_size=args.batch_size)
            print('Test loss: {}'.format(validation_loss))
            #validation_loss = lm.get_loss(test_lines, batch_size=1)
            #print('Test loss: {}'.format(validation_loss))
            print('PPL: {}'.format(lm.get_word_level_perplexity(test_lines, add_bos_token=False)))

    # Estimate word-level surprisal values for sentences
    if args.do_eval:
        lm.model.eval()

        if args.fpath is not None:
            sents = load_data(args.fpath)
        else:
            sents = ["The dogs under the tree are barking.", "The dogs under the tree is barking.",
                    "The keys to the cabinet are on the table.", "The keys to the cabinet is on the table.",]

        print('sentence_id\ttoken_id\ttoken\tsurprisal')

        for i, sent in enumerate(sents):
            if args.pretokenized:
                words = sent.strip().split()
                stimulus = sent.strip()
            else:
                words = nltk.word_tokenize(sent.strip())
                stimulus = ' '.join(words)

            tokens = lm.tokenizer.tokenize(stimulus)
            with torch.no_grad():
                surprisals = lm.get_surprisals(tokens, add_bos_token=True)

            index = 0
            for j, word in enumerate(words):
                w_str = ''
                w_surprisal = 0
                while index < len(tokens) and w_str != word:
                    token_str = tokens[index]
                    if token_str.startswith(lm.w_boundary_char):
                        w_str += token_str[1:]
                    else:
                        w_str += token_str
                    w_surprisal += surprisals[index]

                    index += 1

                print('{}\t{}\t{}\t{}'.format(i+1, j+1, word, w_surprisal))
