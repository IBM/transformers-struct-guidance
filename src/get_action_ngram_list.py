from argparse import ArgumentParser

def is_nt(token):
    if token.startswith('NT(') and token.endswith(')'):
        flag = True
    else:
        flag = False
    return flag


def is_reduce(token):
    if token == 'REDUCE()':
        flag = True
    else:
        flag = False
    return flag


def update_ngram_freq_dict(path, action_ngram_freq_dict):
    count = 0
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            action_seq = []
            for i, token in enumerate(tokens):
                if is_nt(token) or is_reduce(token):
                    action_seq.append(token)
                else:
                    if action_seq != []:
                        action_ngram = '_'.join(action_seq)
                        if action_ngram not in action_ngram_freq_dict:
                            action_ngram_freq_dict[action_ngram] = 1
                        else:
                            action_ngram_freq_dict[action_ngram] += 1
                    action_seq = []

            count += 1
    return count


parser = ArgumentParser()
parser.add_argument('-f', '--fpaths', nargs='+', default=['bllip-lg_train_gen.oracle', 'bllip-lg_dev_gen.oracle'], help="List of paths to parsing oracle files.")
parser.add_argument('-o', '--output', default='bllip-lg_action_ngram_list.txt', help='Path to save the action ngram list.')
args = parser.parse_args()

paths = args.fpaths
output_path = args.output

action_ngram_freq_dict = {}

sent_count = 0
for path in paths:
    sent_count += update_ngram_freq_dict(path, action_ngram_freq_dict)
print('{} sentences in total.'.format(sent_count))

action_ngram_list = [(k, v) for k, v in action_ngram_freq_dict.items()]

# Sort the action ngram list based on frequency
action_ngram_list_sorted = sorted(action_ngram_list, key = lambda x: x[1], reverse=True)

with open(output_path, 'w') as f:
    for action_ngram, freq in action_ngram_list_sorted:
        f.write(action_ngram+'\t'+str(freq)+'\n')

print(len(action_ngram_list), 'action ngrams.')