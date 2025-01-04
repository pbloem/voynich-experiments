import fire

from collections import Counter
import math, random

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tqdm import trange, tqdm

WORDS = { # Words to highlight in the untagged plots
    'alice (tokenized)' : {'and', 'two', 'alice', 'queen', 'or', 'into', 'one', 'three', 'four'},
}

TFILES = { #tagged files
    'frauenfrage' : './frauenfrage.tagged.txt',
    'musulmani' : './musulmani.tagged.txt',
    'wonderbouw': './wonderbouw.tagged.txt',
    'alice': './alice.tagged.txt',
    'poesias': './poesias.tagged.txt',
    # 'dialogo': './dialogo.tagged.txt',
}

UFILES = { #untagged files
    'vms': '../data/eva.takahashi.txt',
    'vms (tokenized)' : './vms.tokenized.txt',
    'alice (tokenized)' : 'alice.tokenized.txt',
}

TAGS = ['noun', 'pronoun', 'verb', 'adjective', 'adverb', 'det', 'numeral', 'other']

def tokenize(corpus, num_tokens, verbose=False):

    corpus = list(corpus)
    tokens = set(corpus)

    with tqdm(total=num_tokens) as pbar:
        pbar.update(len(tokens))
        while len(tokens) < num_tokens and len(corpus) > 1:
            # find the most frequent bigram
            ctr = Counter()
            for i in range(len(corpus)-1):
                bigram = tuple(corpus[i:i+2])
                ctr[bigram] += 1

            t1, t2  = ctr.most_common(1)[0][0]

            # if verbose: print('most common token', t1, t2)

            newtoken = t1 + t2

            tokens.add(newtoken)
            pbar.update(1)

            nwcorpus = []
            i = 0
            while i < len(corpus):
                bigram = tuple(corpus[i:i+2])
                if bigram == (t1, t2):
                    nwcorpus.append(newtoken)
                    i += 2
                else:
                    nwcorpus.append(bigram[0])
                    i += 1

            corpus = nwcorpus

    return corpus

def test():

    corpus = ['a', 'a', 'b', 'c', 'd', 'a', 'a', 'a', 'b', 'c', 'a', 'a', 'b', 'c']

    corpus = tokenize(corpus, 7, True)

    print(corpus)

def ri(x):
    return int(round(x))

class TagModel():

    def __init__(self, corpus=None):
        self.tags = {}  # Maps a word to a Counter over the tags the word is tagged with

        if corpus is not None:
            self.read(corpus)

    def read(self, corpus):
        """
        Observe all tokens in a corpus.
        :param corpus: An iterable of (word, tag) pairs
        :return:
        """
        for word, tag in corpus:
            self.add(word, tag)

    def known(self, word):
        return word in self.tags

    def add(self, word, tag):

       if not self.known(word):
           self.tags[word] = Counter()
       self.tags[word][tag] += 1

    def tag_dist(self, word):
        """
        :param word:
        :return: The relative frequency of tags for the given word as a tuple of floats, in the order indicated by `TAGS`.
        """
        ctr = self.tags[word]
        return [ctr[tag] / ctr.total() for tag in TAGS]

class LevelModel():
    """
    Collects the level stats and

    -- The implementation is less efficient, but simpler than the Java version.
    """

    fnames = ['relative frequency', 'sigma', 'sigma (normalized)', 'C']

    def __init__(self, corpus=None):
        self.distances = {}  # Maps a word to the sequence of distances for that word

        self.sumd = {} # sum of the distances so far per word
        self.sumdsq = {} # sum of the squared distances so far

        self.freqs = Counter()  # Word frequencies
        self.lastpos = {}  # Maps a word to the last position at which it was observed
        self.pos = 0  # Position in the text (incremented for every token added)

        if corpus is not None:
            self.read(corpus)

    def read(self, corpus):
        """
        Observe all tokens in a corpus.
        :param corpus: An iterable of (word, tag) pairs or of strings
        :return:
        """
        for token in corpus:
            if type(token) is str:
                self.add(token) # corpus contains only words
            elif type(token) is tuple or type(token) is list:
                self.add(token[0]) # corpus contains tuples (tagged words)
            else:
                raise Exception(f'{token} has type {type(token)}. Should be tuple, list or str.')

    def known(self, word):
        return word in self.freqs

    def add(self, word):
        """
        Observe a token (a word, tag pair)

        :param word:
        :param tag:
        :return:
        """
        if not self.known(word):
            self.distances[word] = []
            self.sumd[word]   = 0
            self.sumdsq[word] = 0
        else:
            d = self.pos - self.lastpos[word]
            self.distances[word].append(d)
            self.sumd[word] += d
            self.sumdsq[word] += d * d

        self.freqs[word] += 1
        self.lastpos[word] = self.pos

        self.pos += 1

    def freq(self, word):
        return self.freqs[word]

    def prob(self, word):
        return self.freqs[word] / self.freqs.total()

    def sigma(self, word):

        if not self.known(word) or self.freq(word) < 2:
            return float('NaN')

        # avg = sum(self.distances[word]) / len(self.distances[word])
        # avg_sq = sum(d ** 2 for d in self.distances[word]) / len(self.distances[word])

        avg = self.sumd[word] / len(self.distances[word])
        avg_sq = self.sumdsq[word] / len(self.distances[word])

        sd = math.sqrt(avg_sq - avg ** 2)

        return sd / avg

    def signorm(self, word):
        """
        Normalized sigma score.

        :param word:
        :return:
        """

        return self.sigma(word) / math.sqrt(1.0 - self.prob(word));

    def c(self, word):

        n = self.freq(word)
        mean =  (2.0 * n - 1) / (2.0 * n + 2)
        stdv =  1 / (math.sqrt(n) * (1.0 + 2.8 * (n ** -0.865)))

        return (self.signorm(word) - mean) / stdv

    def features(self, word):
        """
        Returns a "feature vector" for the given word (prob, sig, signorm, c)
        :param word:
        :return: a tuple containing four floats.
        """
        return self.prob(word), self.sigma(word), self.signorm(word), self.c(word)


def context(infile, tagged=False, context=5, which_feature=3, num_plot=50, words=None, ylim=None):
    if words is not None and type(words) is str:
         words = words.split(',')

    name = infile.split('.')[0]

    with open(infile, 'r') as file:
        all = file.read()

    if tagged:
        corpus = [t.split('.')[0] for t in all.split()]
    else:
        corpus= all.split()

    lmodel = LevelModel(corpus)
    print(f'Created level model from {infile}.' )


    mid = context // 2

    def feat(i, mod):
        if i == 0: return mod.prob
        if i == 1: return mod.sigma
        if i == 2: return mod.signorm
        if i == 3: return mod.c

    top_words = [w for w, f in lmodel.freqs.most_common(num_plot)] if words is None else words
    tw_set = set(top_words)

    results = {}

    # Collect stats
    for fr in trange(len(corpus) - context):
        window = corpus[fr:fr + context]
        word = window[mid]

        if word in top_words:

            values = [feat(which_feature, lmodel)(w) for w in window]
            values = np.asarray(values)

            if word not in results:
                results[word] = []
            results[word].append(values[None, :])

    fig, axes = plt.subplots(nrows=len(top_words), ncols=1, figsize=(8, len(top_words)))
    for i, word in enumerate(top_words):

        windows = np.concatenate(results[word], axis=0)

        ax = axes[i]

        ax.plot(np.arange(context), np.nanmean(windows, axis=0))
        ax.errorbar(np.arange(context), np.nanmean(windows, axis=0), yerr=np.nanstd(windows, axis=0))
        # -- NB: For some words in the context the features will be NaN, so we use nanmean and nanstd

        if ylim is not None:
            ax.set_ylim(*ylim)

        clean(ax)

        #plot the word next to the graph
        props = dict(facecolor='white', linewidth=0, edgecolor='none')  # bbox features
        ax.text(-0.3, 0.5, f'{word} ({lmodel.freq(word)})', transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig(f'{name}-context.png')

    print('done.')

def go(min_freq=10, which_features=(2,3), label=None):

    tmodels = {}
    lmodels = {}
    for name, filestr in TFILES.items():
        with open(filestr, 'r') as file:
            all = file.read()

        corpus = [t.split('.') for t in all.split()]

        tmodels[name] = TagModel(corpus)
        lmodels[name] = LevelModel(corpus)

    for name, filestr in UFILES.items():
        with open(filestr, 'r') as file:
            all = file.read()

        corpus = all.split()
        lmodels[name] = LevelModel(corpus)

    print('Files loaded.')

    nr, nc = len(TFILES) + 1, max(len(TAGS), len(UFILES))
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*8, nr*8))

    # Plot the language data
    for i, name in enumerate(TFILES.keys()): # over datasets

        lmodel, tmodel = lmodels[name], tmodels[name]

        words = [w for w in lmodel.freqs.keys() if lmodel.freqs[w] >= min_freq]

        features = [lmodel.features(word) for word in words]
        features = np.asarray(features)

        # print(f'feature extent {name}')
        # for k in range(features.shape[1]):
        #     print(k, np.min(features[:, k]), np.max(features[:, k]))

        x, y = features[:, which_features[0]], features[:, which_features[1]]
        xl, yl = lmodel.fnames[which_features[0]], lmodel.fnames[which_features[1]]

        for j, tagname in enumerate(TAGS):

            ax = axes[i, j]
            if i == 0:
                ax.set_title(tagname)
            if j == 0:
                ax.set_ylabel(yl)
                ax.set_title(f'{name} ({ri(lmodel.freqs.total()/1000)}k)', loc='left')

            tprob = [tmodel.tag_dist(w)[j] for w in words]

            ax.scatter(x, y, c=tprob, cmap='Reds', s=3, vmin=-0.3, vmax=1)
            # -- the vmin is slightly below 0, so that points with prob=0.0 show up.
            clean(ax)
            ax.set_yscale('log')
            ax.set_xscale('log')

            if label is not None:
                for iw, word in enumerate(words):
                    if tprob[iw] > 0.4:
                        if random.random() < label:
                            ax.annotate(word, (x[iw], y[iw]), size=6)

    # Plot the untagged files


    for j, name in enumerate(UFILES.keys()):

        words = [w for w in lmodels[name].freqs.keys() if lmodels[name].freqs[w] >= min_freq]

        features = [lmodels[name].features(word) for word in words]
        features = np.asarray(features)

        # print(f'feature extent {name}')
        # for k in range(features.shape[1]):
        #     print(k, np.min(features[:, k]), np.max(features[:, k]))

        x, y = features[:, which_features[0]], features[:, which_features[1]]
        xl, yl = lmodel.fnames[which_features[0]], lmodel.fnames[which_features[1]]

        ax = axes[len(TFILES), j] # bottom row, jth column
        ax.set_xlabel(xl)

        if j == 0:
            ax.set_ylabel(yl)

        ax.set_title(f'{name} ({ri(lmodels[name].freqs.total()/1000)}k)', loc='left')

        ax.scatter(x, y, s=3, c='#ccc')

        # -- the vmin is slightly below 0, so that points with prob=0.0 show up.
        clean(ax)
        ax.set_yscale('log')
        ax.set_xscale('log')


        if label:
            for iw, word in enumerate(words):
                if name not in WORDS or word in WORDS[name]:
                    if random.random() < label:
                        ax.annotate(word, (x[iw], y[iw]), size=6)


    for j in range(len(UFILES), nc): # remov ethe remaining plots in the bottom row.
        fig.delaxes(axes[len(TFILES), j])

    plt.savefig('scatter.png')
    print('done.')


def clean(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def ca(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off', top='off', right='off', left='off',
        labelbottom='off', labelleft='off')  # labels along the bottom edge are off


def center(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        labelbottom='off', labelleft='off')

    ax.set_aspect('equal')

if __name__ == '__main__':
    fire.Fire()