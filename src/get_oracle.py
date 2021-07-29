import argparse


def is_valid_action_sequence(action_sequence):
    flag = True
    for k, action in enumerate(action_sequence):
        if action == "REDUCE":
            if k <= 1:
                flag = False
                break
            if action_sequence[k-1].startswith('NT('):
                flag = False
                break
    return flag


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')


def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)
    return ''.join(output)


def get_tags_tokens_lowercase(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions_and_terms(line, is_generative):
    output_actions = []
    output_terms = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                terminal = get_between_brackets(line_strip, i)
                terminal_split = terminal.split()
                assert len(terminal_split) == 2 # each terminal contains a POS tag and word
                token = terminal_split[1]
                output_terms.append(token)
                if is_generative:
                    # generative parsing
                    output_actions.append(token)
                else:
                    # discriminative parsing
                    output_actions += ['SHIFT']
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
            output_actions.append('REDUCE()')
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != ')' and line_strip[i] != '(':
                i += 1
    assert i == max_idx
    return output_actions, output_terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,
                        help='File path to the bracketed constituency trees.')
    parser.add_argument('--gen', action='store_true',
                        help='Get oracle action sequences in the generative mode.')
    parser.add_argument('--root', action='store_true',
                        help='Output root symbol at the beginning of each action sequence.')
    args = parser.parse_args()

    f = open(args.fpath, 'r')
    lines = f.readlines()
    f.close()

    line_ctr = 0
    # get the oracle action sequences for the input file
    for line in lines:
        line_ctr += 1
        # assert that the parenthesis are balanced
        if line.count('(') != line.count(')'):
            raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr))
        output_actions, output_terms = get_actions_and_terms(line, is_generative=args.gen)

        if not is_valid_action_sequence(output_actions):
            continue

        if len(output_actions) > 500:
            continue

        if args.gen:
            if args.root:
                print('[START]', end=' ')
            print(' '.join(output_actions))
        else:
            print(' '.join(output_terms) + '\t' + ' '.join(output_actions))


if __name__ == "__main__":
    main()
