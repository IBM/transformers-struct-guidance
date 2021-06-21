import numpy as np
import os


def load_dataset(tokenizer, path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    #raw_text = '<|endoftext|>'.join(lines) + '<|endoftext|>'
    raw_text = '\n'.join(lines) + '<|endoftext|>'

    token_chunks = []
    tokens = tokenizer.tokenize(raw_text)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.
    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(len(chunk) for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + len(chunks[i]))
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]
