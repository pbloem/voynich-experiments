import fire, math

import numpy as np


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""
Experiments with unsupervised tokenization
"""

class Trie():

    class Node():

        def __init__(self):
            self.children = {}
            self.f = 0

        def add(self, ngram):
            self.f += 1
            if len(ngram) == 0:
                return

            char, rest = ngram[0], ngram[1:]

            if char not in self.children:
                self.children[char] = Trie.Node()

            self.children[char].add(rest)

        def freq(self, ngram):
            if len(ngram) == 0:
                return self.f

            char, rest = ngram[0], ngram[1:]
            if char not in self.children:
                return 0

            return self.children[char].freq(rest)

    """
    A trie: a tree datastructure used to count n-gram frequencies.
    """
    def __init__(self, ngrams=None):
        """
        :param ngrams: An iterable of ngrams to store in the trie.
        """
        self.root = Trie.Node()
        self.chars = set()

        if ngrams is not None:
            for ngram in ngrams:
                self.add(ngram)


    def add(self, ngram):
        self.root.add(ngram)
        self.chars.update(ngram)

    def freq(self, ngram):
        return self.root.freq(ngram)

    @staticmethod
    def from_corpus(corpus, n=3):
        return Trie(corpus[i:i+n] for i in range(len(corpus)-n+1))

    def prob(self, char:str, cond:str, smoothing=1e-6, nchars=None, log=False):
        """
        Returns the probability of observing the character `char`, conditional on
        having observerd the ngram `cond` preceding it.

        :param char: a single-character string
        :param cond: a string
        :param nchars: The total number of characters. If None, the total number
            observed by the trie is used.
        :return:
        """
        if nchars is None: nchars = len(self.chars)

        numerator = self.freq(cond+char) + smoothing
        denominator = self.freq(cond) + smoothing * nchars

        if log:
            return np.log(numerator) - np.log(denominator)
        return numerator / denominator

    def entropy(self, ngram:str, smoothing=1e-4, chars=None):
        """
        Returns the entropy (under the distribution modeled by this trie) of the
        probability conditioned by the given n-gram. That is, if the ngram is `abc`,
        this function returns the entropy of the probability distribution p(?|a,b,c).

        If the max order of the trie is n, then the provided ngram should not be
        longer than n-1.

        :param ngram: An ngram to condition on
        :param chars: The set of all characters. If None, then the set of all characters
        observed by the Trie is used.
        :return:
        """

        if chars is None: chars = self.chars

        ent = 0.0
        for c in chars:
            lprob = self.prob(c, ngram, smoothing=smoothing, log=True)
            ent -= np.exp(lprob) * lprob

        return ent

def tokenize(infile,                    # input: text file
             outfile='tokenized.txt',   # output: text file,
             tagged=False,              # whether the input file is tagged
             rm_whitespace=True,        # Whether to remove the white space
             n=3,                       # The n-gram order to use
             smoothing=1e-8,            # Parameter for Laplace-smoothing the probabilities
             threshold=1.0,
            ):
    """
    Tokenizes a given corpus. Returns the corpus with discovered tokens separated by whitespace. If the whitespace in
    the original is kept (as single space between words) these are replaced by underscores.
    """
    with open(infile, 'r') as file:
        all = file.read()

    if tagged:
        tokens = [token.split('.')[0] for token in all.split()]
    else:
        tokens = all.split()

    joinchar = '' if rm_whitespace else '_'
    corpus = joinchar.join(tokens)

    ## `corpus` is now a single sequence of characters.

    forward, backward = Trie.from_corpus(corpus, n=n), Trie.from_corpus(corpus[::-1], n=n)
    print('Created tries.')

    entropies = [entropy(i, corpus, forward, backward, n) for i in range(len(corpus))]
    entropies = np.asarray(entropies)

    mean, std = np.mean(entropies), np.std(entropies)
    print(mean, std)

    plt.hist(entropies, bins=100)
    plt.savefig('entropies.hist.png')

    l = 50
    r = 5_000, 5_000 + l

    plt.figure(figsize=(l, 4))
    plt.bar(np.arange(l), entropies[r[0]:r[1]], width=0.3)
    plt.xticks(np.arange(l)-0.5, corpus[r[0]:r[1]])
    plt.axhline(mean, linestyle='-')
    plt.axhline(mean + std, linestyle=':')

    plt.savefig('entropies.bars.png')

    tokens = []
    lastbreak = 0 # first character after the last break
    for i in range(len(corpus)):
        ent = entropies[i]
        if ent > mean + threshold * std: # new break between i and i+1
            tokens.append(corpus[lastbreak:i+1])
            lastbreak = i+1

    tokens.append(corpus[lastbreak:])

    with open(outfile, 'w') as file:
        file.write(' '.join(tokens))

def entropy(i, corpus, forward : Trie, backward : Trie, n=3):
    """
    Compute the entropy (forward and backward) across the gap between position i and i+1 in corpus
    :param i:
    :param corpus:
    :param n:
    :return:
    """

    # Check if there should be a break after position i
    before, after = corpus[max(0, i - n + 2):i + 1], corpus[i + 1:i + n]
    # -- We are analyzing the break between i and i+1. This means that we want the
    #    ngram up to and including i for the forward entropy, and the ngram from
    #    and including i+1 for the backward entropy
    if i > n - 1 and i < len(corpus) - n:
        assert len(before) == len(after) == n - 1, f'{before}, {after}, {n - 1}'
    assert before + after == corpus[max(0, i - n + 2):i + n], f'{i=}, {before}, {after}, {corpus[max(0, i - n + 2):i + n]}'

    entforward, entbackward = forward.entropy(before), backward.entropy(after[::-1])

    return min(entforward, entbackward)
    # -- We take the minimal entropy of the two. That is, the maximal predictability. We only
    #    insert a boundary between i and i_+1 if i+1 is poorly predictable from the preceding
    #    characters _and_ i is poorly predictable from the following characters.


def test(n=3):

    corpus = 'aaabcabcbaabaaaabcbcbcaabaxyz'
    trie = Trie.from_corpus(corpus)

    for ngram in ['aaa', 'xyz', 'yz', 'abc', 'aa']:
        print(ngram, trie.freq(ngram))

    print(np.exp(trie.prob('a', 'aa', smoothing=0.0, log=True)))

    corpus = 'aaabaacaad'
    trie = Trie.from_corpus(corpus)
    print(trie.entropy('aa', smoothing=1e-8) / np.log(2))
    print(trie.entropy('ab', smoothing=1e-8) / np.log(2))

if __name__ == '__main__':
    fire.Fire()
