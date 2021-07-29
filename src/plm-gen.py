from ipdb import set_trace
import os
import sys
import re
import time
import argparse
import random
import numpy as np
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
import torch
import functools
import utils
print = functools.partial(print, flush=True)

class PLM:
    def __init__(self, is_random_init, NT_CATS, REDUCE, ROOT, device='cuda', model_name='gpt2', cache_dir='pretrained/gpt2'):
        # Load pretrained tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            self.config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(self.config).to(device)
        else:
            print('Initialize with pretrained weights', file=sys.stderr)
            self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)

        # specify special tokens
        self.SPECIAL_BRACKETS = ['-LRB-', '-RRB-', '-LCB-', '-RCB-']

        # specify GEN actions
        self.GEN_VOCAB = self.tokenizer.convert_ids_to_tokens(range(len(self.tokenizer)-1)) + self.SPECIAL_BRACKETS  # '<|endoftext|>' not included

        # specify non-GEN parsing actions
        self.NT_CATS = NT_CATS
        self.REDUCE = REDUCE
        self.ROOT = ROOT

        self.NT_ACTIONS = ["NT({})".format(cat) for cat in self.NT_CATS]
        self.NT_ACTIONS_SET = set(self.NT_ACTIONS)
        self.NT_ACTIONS2NT_CAT = dict([["NT({})".format(cat), cat] for cat in self.NT_CATS])
        self.ACTIONS_SET = set(self.NT_ACTIONS + [self.REDUCE]) # the set of non-terminal actions and reduce

        self.w_boundary_char = b'\xc4\xa0'.decode()

        self.a2str = {}
        for cat in self.NT_CATS:
            a = "NT({})".format(cat)
            self.a2str[a] = '('+cat

        self.num_added_toks = self.tokenizer.add_tokens(self.SPECIAL_BRACKETS + self.NT_ACTIONS + [self.REDUCE, self.ROOT])
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.GEN_ids = self.tokenizer.convert_tokens_to_ids(self.GEN_VOCAB)
        self.NT_ids = self.tokenizer.convert_tokens_to_ids(self.NT_ACTIONS)
        self.REDUCE_id = self.tokenizer.convert_tokens_to_ids(self.REDUCE)        

    def tokenize_batch(self, line_batch):
        # Tokenize a batch of sequences. Add prefix space.
        words_batch = [line.strip().split() for line in line_batch]
        tokens_batch = [[token for word in words for token in self.tokenizer.tokenize(word, add_prefix_space=True)] for words in words_batch]
        return tokens_batch

    def is_valid_action(self, action, nt_count, reduce_count, prev_action, buffer_size, max_open_nt=50):
        '''
        Given a parsing state, check if an action is valid or not.
        buffer_size set to -1 if sampling action sequences
        '''
        flag = True
        if action in self.NT_ACTIONS_SET:
            if (buffer_size == 0) or (nt_count - reduce_count > max_open_nt):
                flag = False
        elif action == REDUCE:
            if (prev_action in self.NT_ACTIONS_SET) or (buffer_size > 0 and nt_count-reduce_count == 1) or prev_action == ROOT:
                flag = False
        else:
            if (buffer_size == 0) or (prev_action == ROOT):
                flag = False
        return flag

    def get_sample(self, prefix, top_k, add_structured_mask):
        '''
        Given a prefix of a generative parsing action sequence, sample a continuation
        from the model, subject to the constraints of valid actions.
        Return a bracketed tree string.
        prefix: 
        '''
        nt_count = 0
        reduce_count = 0
        tree_str = ''

        prefix_tokens = self.tokenizer.tokenize(prefix)
        prefix_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
        prev_token = prefix_tokens[-1]

        while (nt_count-reduce_count != 0 or nt_count == 0) and nt_count < 40:
            input_ids = torch.tensor(prefix_ids).unsqueeze(0).to(device)
            if add_structured_mask:
                attention_mask = get_attention_mask_from_actions(prefix_tokens)
                prediction_scores = self.model(input_ids, attention_mask=attention_mask)[0]
            else:
                prediction_scores = self.model(input_ids)[0] # batch size = 1

            while True:
                token_id = sample_from_scores(prediction_scores[:, -1], top_k=top_k)[0]
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                if self.is_valid_action(token, nt_count, reduce_count, prev_token, buffer_size=-1, max_open_nt=50):
                    break
            
            prefix_tokens.append(token)
            prefix_ids.append(token_id)

            if token in self.NT_ACTIONS_SET:
                if prev_token not in self.NT_ACTIONS_SET and nt_count > 0:
                    tree_str += ' ' + self.a2str[token] + ' '
                else:
                    tree_str += self.a2str[token] + ' '
                nt_count += 1
            elif token == REDUCE:
                tree_str += ')'
                reduce_count += 1
            else:
                if token.startswith(self.w_boundary_char):
                    token_new = token.replace(self.w_boundary_char, '')
                    if prev_token in self.NT_ACTIONS_SET:
                        tree_str += token_new
                    else:
                        tree_str += ' ' + token_new
                else:
                    tree_str += token

            prev_token = token
        return tree_str


    def get_actions_and_ids(self, nt_count, reduce_count, buffer_size, prev_action, max_open_nt=50):
        '''
        Given parameters of a parser state, return a list of all possible valid actions and a list of their ids.
        '''
        valid_actions = []
        valid_action_ids = []

        # NT actions
        if (buffer_size < 1) or (nt_count - reduce_count > max_open_nt):
            pass
        else:
            valid_actions += self.NT_ACTIONS
            valid_action_ids += self.NT_ids

        # REDUCE action
        if (prev_action in self.NT_ACTIONS_SET) or (buffer_size > 0 and nt_count-reduce_count == 1) or prev_action == ROOT:
            pass
        else:
            valid_actions += [REDUCE]
            valid_action_ids += [self.REDUCE_id]
        
        # GEN action
        if buffer_size < 1 or prev_action == ROOT:
            pass
        else:
            valid_actions += self.GEN_VOCAB
            valid_actions += self.GEN_ids

        return valid_actions, valid_action_ids    

    def select_valid_actions(self, actions, nt_count, reduce_count, buffer_size, prev_action, max_open_nt=50):
        '''
        Given parameters of a parser state as well as a subset of all possible actions, select the valid actions.
        '''
        valid_actions = []
        for action in actions:
            flag = self.is_valid_action(action, nt_count=nt_count, reduce_count=reduce_count, buffer_size=buffer_size, prev_action=prev_action, max_open_nt=max_open_nt)
            if flag:
                valid_actions.append(action)
        return valid_actions 

    def add_gen_vocab(self, valid_actions):
        if valid_actions[-1] in self.ACTIONS_SET:
            action_all = valid_actions
        else:
            action_all = valid_actions[:-1] + self.GEN_VOCAB
        return action_all      

    def get_adist(self, scores, valid_actions, index=-1):
        '''
        Given logit scores and a list of valid actions, return normalized probability distribution over valid actions
        and a dictionary mapping specific action tokens to corresponding position index in the valid action list.
        '''
        if valid_actions[-1] in self.ACTIONS_SET:
            action_all = valid_actions
        else:
            action_all = valid_actions[:-1] + self.GEN_VOCAB
            
        valid_action_ids = self.tokenizer.convert_tokens_to_ids(action_all)
        action_scores = scores[:, index, valid_action_ids]
        adist = log_softmax(action_scores.squeeze())
        a2idx = dict([[a, idx] for idx, a in enumerate(action_all)])
        return adist, a2idx    

    def decode_tree_str(self, prefix_actions):
        '''
        Given a sequence of actions, return a bracketed tree string.
        '''
        tree_str = ""
        for a in prefix_actions:
            if a == REDUCE:
                tree_str += ")"
            elif a in self.NT_ACTIONS_SET:
                a_cat = self.NT_ACTIONS2NT_CAT[a]
                tree_str += " (" + a_cat
            else:
                if a.startswith(self.w_boundary_char):
                    term_new = a.replace(self.w_boundary_char, '')
                    tree_str += ' ' + term_new
                else:
                    tree_str += a
        return tree_str

    def get_adist_batch(self, pq_this, token, buffer_size, add_structured_mask, batch_size=50):
        '''
        Given a list of incremental parser states, get the probability distribution
        of valid incoming actions for each parser state. Perform batched computations.
        '''
        pq_this_adist = None

        # compute total number of batches needed
        pq_this_len = len(pq_this)
        if pq_this_len % batch_size == 0:
            num_batches = pq_this_len // batch_size
        else:
            num_batches = pq_this_len // batch_size + 1

        for i in range(num_batches):
            start_idx = batch_size*i
            end_idx = batch_size*(i+1)
            #set_trace(context=10)
            #print(start_idx, end_idx)
            pq_this_batch = pq_this[start_idx:end_idx]

            prefix_max_len = np.max([len(p_this.prefix_actions) for p_this in pq_this_batch])
            
            input_ids_batch = torch.tensor([self.tokenizer.convert_tokens_to_ids(p_this.prefix_actions + ['#' for _ in range(prefix_max_len-len(p_this.prefix_actions))]) for p_this in pq_this_batch]).to(device)

            if add_structured_mask:
                attention_mask_batch = torch.ones(len(pq_this_batch), 12, prefix_max_len, prefix_max_len).to(device)
                for b_idx, p_this in enumerate(pq_this_batch):
                    attention_mask_batch[b_idx, :, :, :] = get_attention_mask_from_actions(p_this.prefix_actions, max_len=prefix_max_len)
            else:
                attention_mask_batch = torch.ones(len(pq_this_batch), 12, prefix_max_len, prefix_max_len).to(device)
                for b_idx, p_this in enumerate(pq_this_batch):
                    attention_mask_batch[b_idx, :, :, len(p_this.prefix_actions):] = 0           

            prediction_scores_batch = self.model(input_ids_batch, attention_mask=attention_mask_batch)[0]

            actions = self.NT_ACTIONS + [REDUCE] + [token]

            pq_this_all_valid_actions_batch = []
            pq_this_all_valid_action_ids_batch = []
            for p_this in pq_this[batch_size*i:batch_size*(i+1)]:
                valid_actions, valid_action_ids = self.get_actions_and_ids(p_this.nt_count, p_this.reduce_count, buffer_size, p_this.prev_a, max_open_nt=50)
                pq_this_all_valid_actions_batch.append(valid_actions)
                pq_this_all_valid_action_ids_batch.append(valid_action_ids)

            scores_index_mask = torch.zeros(len(pq_this_batch), len(self.tokenizer)).to(device)

            indice_dim_0 = []
            indice_dim_1 = []
            for p_index, valid_action_ids in enumerate(pq_this_all_valid_action_ids_batch):
                indice_dim_0 += [p_index for _ in range(len(valid_action_ids))]
                indice_dim_1 += valid_action_ids
            indice_dim_0 = torch.tensor(indice_dim_0).to(device)
            indice_dim_1 = torch.tensor(indice_dim_1).to(device)

            scores_index_mask[indice_dim_0, indice_dim_1] = 1

            pq_this_action_scores_batch = prediction_scores_batch[torch.arange(len(pq_this_batch)), [len(p_this.prefix_actions) - 1 for p_this in pq_this_batch], :]
            pq_this_action_scores_batch.masked_fill(scores_index_mask == 0, -np.inf)
            if pq_this_adist == None:
                pq_this_adist = log_softmax(pq_this_action_scores_batch).detach()
            else:
                pq_this_adist = torch.cat((pq_this_adist, log_softmax(pq_this_action_scores_batch).detach()), 0)
        return pq_this_adist


    def get_surprisals_with_beam_search(self, sent, add_structured_mask, beam_size=100, word_beam_size=10, fast_track_size=5, batched=True, debug=False):
        '''
        Estimate surprisals -log2(P(x_t|x_1 ... x_{t-1})) at subword token level.
        Return a list of subword tokens and surprisals.
        '''
        tokens = self.tokenizer.tokenize(sent)
        idxs = self.tokenizer.convert_tokens_to_ids(tokens)

        prefix = ROOT
        prefix_tokens = self.tokenizer.tokenize(prefix)
        prefix_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)

        surprisals = []
        log_probs = []

        pq_this = []
        init = ParserState(prefix_actions=[ROOT], score=0, nt_count=0, reduce_count=0, prev_a=ROOT)
        pq_this.append(init)

        for k, token in enumerate(tokens):
            print('Token index: {} {}'.format(k, token), file=sys.stderr)
            pq_next = []

            while len(pq_next) < beam_size:
                fringe = []

                if batched:
                    start_time = time.time()
                
                    if k <= 80:
                        eval_batch_size = 100
                    else:
                        eval_batch_size = 50

                    pq_this_adist_batch = self.get_adist_batch(pq_this, token, buffer_size=len(tokens)-k, add_structured_mask=add_structured_mask, batch_size=eval_batch_size)

                start_time = time.time()

                for p_index, p_this in enumerate(pq_this):
                    actions = self.NT_ACTIONS + [REDUCE] + [token]
                    buffer_size = len(tokens) - k
                    current_valid_actions = self.select_valid_actions(actions, p_this.nt_count, p_this.reduce_count, buffer_size, p_this.prev_a, max_open_nt=50)
                    
                    if batched:
                        # using batched computation
                        adist = pq_this_adist_batch[p_index]
                    else:
                        # not using batched computation
                        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(p_this.prefix_actions)).unsqueeze(0).to(device)
                        prediction_scores = self.model(input_ids)[0]
                        adist, a2idx = self.get_adist(prediction_scores, current_valid_actions)

                    for action in current_valid_actions:
                        if batched:
                            a_idx = self.tokenizer.convert_tokens_to_ids(action)
                        else:
                            a_idx = a2idx[action]
                        new_score = p_this.score + adist[a_idx].item()
                        new_nt_count = p_this.nt_count + 1 if action in self.NT_ACTIONS_SET else p_this.nt_count
                        new_reduce_count = p_this.reduce_count + 1 if action == REDUCE else p_this.reduce_count
                        p_state = ParserState(prefix_actions=p_this.prefix_actions+[action], score=new_score, 
                                                nt_count=new_nt_count, reduce_count=new_reduce_count, prev_a=action)
                        fringe.append(p_state)
                

                fringe = prune(fringe, len(fringe))
                fast_track_count = 0
                cut = np.max([len(fringe)-beam_size, 0])

                pq_this_new = []
                for k in range(len(fringe)-1, -1, -1):
                    if k >= cut:
                        if fringe[k].prev_a not in self.ACTIONS_SET:
                            pq_next.append(fringe[k])
                        else:
                            pq_this_new.append(fringe[k])
                    else:
                        if fringe[k].prev_a not in self.ACTIONS_SET and fast_track_count < fast_track_size:
                            pq_next.append(fringe[k])
                            fast_track_count += 1
                pq_this = pq_this_new

                if debug:
                    print("--- %s seconds for sorting parser states---" % (time.time() - start_time))

            pruned_pq_next = prune(pq_next, word_beam_size)

            print("List of partial parses:", file=sys.stderr)
            for beam_index, pstate in enumerate(pruned_pq_next):
                print('{} {:.3f} {}'.format(beam_index, pstate.score, self.decode_tree_str(pstate.prefix_actions)), file=sys.stderr)
            
            # Use log-sum-exp
            log_probs.append(-logsumexp([ps.score for ps in pruned_pq_next])/np.log(2))

            pq_this = pruned_pq_next
        
        for k in range(len(log_probs)):
            if k == 0:
                surprisals.append(log_probs[k])
            else:
                surprisals.append(log_probs[k] - log_probs[k-1])

        for k in range(len(surprisals)):
            print(tokens[k], surprisals[k], file=sys.stderr)

        return tokens, surprisals


    def get_validation_loss(self, dev_lines, add_structured_mask):
        loss_sum = 0
        token_count = 0

        for line in dev_lines:
            words = line.strip().split()
            tokens = [ROOT] + [token for word in words for token in self.tokenizer.tokenize(word, add_prefix_space=True)]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device) # batch size = 1

            if add_structured_mask:
                # play actions on RNNG state machine to get the different states
                # and derive mask values from them
                # size [1, num_heads, from_seq_length, to_seq_length]
                attention_mask = get_attention_mask_from_actions(tokens)

                # Update model
                loss = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)[0].item()
            else:
                loss = self.model(input_ids, labels=input_ids)[0].item()
            loss_sum += loss*(len(tokens)-1)
            token_count += len(tokens) - 1
        return loss_sum/token_count


    def get_batch_loss(self, lines, add_structured_mask, device='cuda'):
        line_batch = [ROOT + ' ' + line for line in lines]
        tokens_batch = self.tokenize_batch(line_batch)

        token_count_batch = [len(tokens) for tokens in tokens_batch]
        batch_max_len = np.max(token_count_batch)

        tokens_padded_batch = [tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

        ids_batch = [self.tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
        input_ids = torch.tensor(ids_batch).to(device)

        label_ids_batch = [self.tokenizer.convert_tokens_to_ids(tokens) + [-100 for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]
        label_ids = torch.tensor(label_ids_batch).to(device)        

        if add_structured_mask:
            attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
            for b_idx, tokens in enumerate(tokens_batch):
                attention_mask[b_idx, :, :, :] = get_attention_mask_from_actions(tokens_batch[b_idx], max_len=batch_max_len)
            loss = self.model(input_ids, labels=label_ids, attention_mask=attention_mask)[0]
        else:
            attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
            for b_idx, tokens in enumerate(tokens_batch):
                attention_mask[b_idx, :, :, len(tokens):] = 0               
            loss = self.model(input_ids, labels=label_ids, attention_mask=attention_mask)[0]

        batch_token_count = np.sum(token_count_batch) - len(tokens_batch) # substract the count since len(tokens)-1 words are counted        
        return loss, batch_token_count


    def get_loss(self, lines, add_structured_mask, batch_size, device='cuda'):
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(lines, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, add_structured_mask=add_structured_mask, device=device)
            total_loss += loss.item()*batch_token_count
            total_token_count += batch_token_count

        return total_loss/total_token_count


    def estimate_word_ppl(self, dev_lines, add_structured_mask):
        loss_sum = 0
        word_count = 0

        for line in dev_lines:
            words = line.strip().split()
            tokens = [ROOT] + [token for word in words for token in self.tokenizer.tokenize(word, add_prefix_space=True)]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device) # batch size = 1

            if add_structured_mask:
                # play actions on RNNG state machine to get the different states
                # and derive mask values from them
                # size [1, num_heads, from_seq_length, to_seq_length]
                attention_mask = get_attention_mask_from_actions(tokens)

                # Update model
                loss = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)[0].item()
            else:
                loss = self.model(input_ids, labels=input_ids)[0].item()
            loss_sum += loss*(len(tokens)-1)
            word_count += len([word for word in words if word not in self.ACTIONS_SET])
        return np.exp(loss_sum/word_count)



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


def sample_from_scores(logits, top_k=50):
    kth_vals, kth_idx = logits.topk(top_k, dim=-1)
    sample_dist = torch.distributions.categorical.Categorical(logits=kth_vals)
    token_idx_new = kth_idx.gather(dim=1, index=sample_dist.sample().unsqueeze(-1)).squeeze(-1)  
    return token_idx_new.tolist() 


class ParserState:
    def __init__(self, prefix_actions, score, nt_count, reduce_count, prev_a):
        self.prefix_actions = prefix_actions
        self.score = score
        self.nt_count = nt_count
        self.reduce_count = reduce_count
        self.prev_a = prev_a


def prune(p_state_list, k):
    if len(p_state_list) == 1:
        return p_state_list
    if k > len(p_state_list):
        k = len(p_state_list)
    score_list = [item.score for item in p_state_list]
    sorted_indice = np.argsort(score_list)[-k:]
    pruned_p_state_list = [p_state_list[i] for i in sorted_indice]
    return pruned_p_state_list


nt_re = re.compile('NT\((.*)\)')


class RNNGMachine():

    def __init__(self):
        self.nt_stack = []
        self.previous_stacks = []
        self.actions = []
        self.composed_nt = None

    def get_valid_actions(self):
        """Return valid actions for this state at test time"""

        valid_actions = []

        # This will expand to all NT(*) actions in train
        valid_actions += ['NT'] 

        if len(self.nt_stack) > 0:
            if len(self.buffer):
                valid_actions.append("GEN")

            if (
                # prohibit closing empty constituent
                not self.actions[-1].startswith("NT(") 
                # prohibit closing top constituent if buffer not empty
                and not len(self.nt_stack) == 1
            ):
                valid_actions.append(REDUCE)

        return valid_actions, []

    def update(self, action):

        if nt_re.match(action):
            label = nt_re.match(action).groups()[0]
            self.nt_stack.append(label)
            self.previous_stacks.append(len(self.actions))

        elif action == REDUCE:    
            
            # specify that start position of the non-terminal phrase to be composed

            assert len(self.nt_stack) 
            self.nt_stack.pop()
            # move stack to containing previous constituent
            if self.previous_stacks:
                self.previous_stacks.pop() 
            if len(self.nt_stack) == 0:
                self.is_closed = True

        elif action == '[START]':
            pass

        # Store action
        self.actions.append(action)


def get_attention_mask_from_actions(tokens, max_len=None):
    '''
    Given a list of actions, it returns the attention head masks for all the
    parser states
    '''

    # select which heads we mask
    buffer_head = args.buffer_head
    stack_head = args.stack_head

    # Start RNNG Machine
    rnng_machine = RNNGMachine()
    if max_len is None:
        # single sentence
        attention_mask = torch.ones(1, 12, len(tokens), len(tokens))
    else:
        # multiple sentence, we need to pad to max_len
        attention_mask = torch.ones(1, 12, max_len, max_len)

    attention_mask = attention_mask.to(device)
    for t, action in enumerate(tokens):
        # update machine 
        rnng_machine.update(action)
        # store state as masks of transformer
        if rnng_machine.previous_stacks:
            # print(rnng_machine.actions[rnng_machine.previous_stacks[-1]:])
            stack_position = rnng_machine.previous_stacks[-1]
            attention_mask[:, buffer_head, t, stack_position:] = 0
            attention_mask[:, stack_head, t, :stack_position] = 0 

    # ensure pad is zero at testing
    if max_len is not None:
        # FIXME: Unsure about masking pattern for element 2
        attention_mask[0, :, :, len(tokens):] = 0 

    return attention_mask


def logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='Path to training data.')
    parser.add_argument('--dev_data', type=str, help='Path to validation data.')
    parser.add_argument('--test_data', type=str, help='Path to test data.')
    parser.add_argument('--fpath', type=str, help='File path to the evaluation materials for estimating surprisals.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--report', type=int, default=1000, help='Frequency of report training status after number of training batches')
    parser.add_argument('--valid_every', type=int, default=None, help='Frequency of validating and saving model parameters after number of training batches')
    parser.add_argument('--sample_every', type=int, default=10000, help='Frequency of generating samples from the model during training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--do_train', action='store_true', help='Whether to train the model')
    parser.add_argument('--do_test', action='store_true', help='Whether to test the model')
    parser.add_argument('--do_eval', action='store_true', help='Whether to use the model for surprisal estimation.')
    parser.add_argument('--model_path', type=str, default=None, help='Name prefix of the model to be trained and saved')
    parser.add_argument('--restore_from', type=str, default=None, help='Path to the trained model checkpoint. Will use the pretrained model if path not specified.')
    parser.add_argument('--beam_size', type=int, default=100, help='Size of action beam')
    parser.add_argument('--word_beam_size', type=int, default=10, help='Size of word beam')
    parser.add_argument('--fast_track_size', type=int, default=5, help='Size of fast track')
    parser.add_argument('--pretokenized', action='store_true', help='Whether stimulus sentences for surprisal estimation are pretokenized or not.')
    parser.add_argument('--batch_size', type=int, default=5, help="Size of a training batch.")
    parser.add_argument('--frequent_validation', action='store_true', help="Add frequent validations across epochs")
    parser.add_argument('--random_init', action='store_true', help="Randomly initialize model parameters.")
    parser.add_argument('--early_stopping_threshold', type=int, default=2, help='Threshold for early stopping.')
    parser.add_argument('--add_structured_mask', action='store_true', help="Use structurally masked attention")
    parser.add_argument('--buffer_head', type=int, help='Specify the index of attention heads for buffer-related structural masks.')
    parser.add_argument('--stack_head', type=int, help='Specify the index of attention heads for stack-related structural masks.')
    parser.add_argument('--add_eos', action='store_true', help="Add ROOT token as the end-of-sequence token.")


    args = parser.parse_args()

    log_softmax = torch.nn.LogSoftmax(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

    # specify non-GEN parsing actions
    NT_CATS = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 
                    'PRN', 'PRT', 'QP', 'RRC', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'UCP', 'VP', 
                    'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X']
    REDUCE = 'REDUCE()'
    ROOT = '[START]'

    plm = PLM(is_random_init=args.random_init, NT_CATS=NT_CATS, REDUCE=REDUCE, ROOT=ROOT, device=device)

    if args.restore_from is not None:
        print('Load parameters from {}'.format(args.restore_from), file=sys.stderr)
        checkpoint = torch.load(args.restore_from)
        plm.model.load_state_dict(checkpoint['model_state_dict'])

    if args.add_structured_mask:
        assert args.buffer_head is not None
        assert args.stack_head is not None
        print('Use structured attention mask: buffer_head {}; stack_head {}'.format(args.buffer_head, args.stack_head), file=sys.stderr)

    
    # Train
    if args.do_train:
        # Path to save the newly trained model
        MODEL_PATH = args.model_path if args.model_path is not None else "plm_pid{}.params".format(os.getpid())

        # Print out training settings
        print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
        print('Learning rate: {}'.format(args.lr), file=sys.stderr)
        print('Model path: {}'.format(MODEL_PATH), file=sys.stderr)

        # Set the learning rate of the optimizer
        optimizer = AdamW(plm.model.parameters(), lr=args.lr)

        # Load train and dev data
        train_data_path = args.train_data
        dev_data_path = args.dev_data
        print("Loading train data from {}".format(train_data_path), file=sys.stderr)
        print("Loading dev data from {}".format(dev_data_path), file=sys.stderr)
        train_lines = load_data(train_data_path)
        dev_lines = load_data(dev_data_path)

        # Add EOS token
        if args.add_eos:
            print('Add EOS token.')
            train_lines = [line + ' ' + ROOT for line in train_lines]

        if args.restore_from is not None:
            plm.model.eval()
            with torch.no_grad():
                best_validation_loss = plm.get_loss(dev_lines, add_structured_mask=args.add_structured_mask, batch_size=args.batch_size)
            plm.model.train()
            print('resume training; validation loss: {}'.format(best_validation_loss))
        else:
            best_validation_loss = np.Inf

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

                loss, _ =plm.get_batch_loss(line_batch, add_structured_mask=args.add_structured_mask, device=device)
                
                loss.backward()
                optimizer.step()

                count += len(line_batch)
                batch_count += 1

                if batch_count > 0 and batch_count % args.report == 0:
                    print('Epoch {:.3f} loss: {}'.format(epoch + count/len(train_lines), loss.item()))

                if batch_count > 0  and batch_count % args.sample_every == 0:
                    plm.model.eval()
                    with torch.no_grad():
                        tree_str = plm.get_sample(prefix=ROOT, top_k=100, add_structured_mask=args.add_structured_mask)
                        print(tree_str)
                    plm.model.train()

                if VALID_EVERY is not None:
                    if batch_count > 0  and batch_count % VALID_EVERY == 0:
                        plm.model.eval()
                        with torch.no_grad():
                            validation_loss = plm.get_loss(dev_lines, add_structured_mask=args.add_structured_mask, batch_size=args.batch_size)
                        print('Epoch {:.3f} validation loss: {}'.format(epoch + count/len(train_lines), validation_loss))

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
                                'add_structured_mask': args.add_structured_mask,
                                'no_improvement_count': early_stopping_counter.counter,
                                'model_state_dict': plm.model.state_dict(),
                                'loss': validation_loss}, MODEL_PATH)
                        plm.model.train()                        


            plm.model.eval()
            with torch.no_grad():
                validation_loss = plm.get_loss(dev_lines, add_structured_mask=args.add_structured_mask, batch_size=args.batch_size)
            print('Epoch', epoch, 'validation loss:', validation_loss)

            is_early_stop = early_stopping_counter.check_stopping_criterion(validation_loss)
            if is_early_stop:
                print('Validation loss increases for {} epochs in a row.'.format(early_stopping_counter.counter))
                print('EARLY STOPPING...')
                sys.exit()
                break

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                print("new best... saving model to {}".format(MODEL_PATH))
                torch.save(
                    {'epoch': epoch,
                    'add_structured_mask': args.add_structured_mask,
                    'no_improvement_count': early_stopping_counter.counter,
                    'model_state_dict': plm.model.state_dict(),
                    'loss': validation_loss}, MODEL_PATH)       

            plm.model.train()
            
    # Test
    if args.do_test:
        plm.model.eval()
        if args.test_data is None:
            raise ValueError('Test data not specified')
        test_data_path = args.test_data
        test_lines = load_data(test_data_path)
        with torch.no_grad():
            test_loss = plm.get_validation_loss(test_lines, add_structured_mask=args.add_structured_mask)
            print('Test loss: {}'.format(test_loss))
            test_loss = plm.get_loss(test_lines, add_structured_mask=args.add_structured_mask, batch_size=args.batch_size)
            print('Test loss: {}'.format(test_loss))
            ppl = plm.estimate_word_ppl(test_lines, add_structured_mask=args.add_structured_mask)
            print('Approximate word PPL: {}'.format(ppl))


    # Estimate token surprisal values for unparsed sentences
    if args.do_eval:
        plm.model.eval()

        if args.fpath is not None:
            sents = load_data(args.fpath)
        else:
            sents = ["The dogs under the tree are barking.", "The dogs under the tree is barking.",
                    "The keys to the cabinet are on the table.", "The keys to the cabinet is on the table.",]

        print('sentence_id\ttoken_id\ttoken\tsurprisal')

        for i, sent in enumerate(sents):
            # add a white space at the sentence initial and pretokenize in PTB format, to be consistent with the format of treebank training data
            if not args.pretokenized:
                words = nltk.word_tokenize(sent)
                stimulus = ' ' + ' '.join(words)
            else:
                words = sent.split()
                stimulus = ' ' + sent

            with torch.no_grad():
                tokens, surprisals = plm.get_surprisals_with_beam_search(stimulus, add_structured_mask=args.add_structured_mask, beam_size=args.beam_size, word_beam_size=args.word_beam_size, fast_track_size=args.fast_track_size, debug=False)

            index = 0
            for j, word in enumerate(words):
                w_str = ''
                w_surprisal = 0
                while index < len(tokens) and w_str != word:
                    token_str = tokens[index]
                    if token_str.startswith(plm.w_boundary_char):
                        w_str += token_str[1:]
                    else:
                        w_str += token_str
                    w_surprisal += surprisals[index]

                    index += 1

                print('{}\t{}\t{}\t{}'.format(i+1, j+1, word, w_surprisal))
