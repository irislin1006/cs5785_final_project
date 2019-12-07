import os
import pickle
import json
import argparse
from collections import Counter
import numpy as np
import re
import pandas as pd
import Constants
from nltk.tokenize import TweetTokenizer

nlp = TweetTokenizer()
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_count = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab_cap(path, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    for i, filename in enumerate(os.listdir(path)):
        if i % 500 == 0:
            print(f"caption: {i}")
        fp = os.path.join(path, filename)
        # print(fp)
        lines = open(fp, 'r').readlines()
        for line in lines:
            text = line.strip().lower()
            tokens = nlp.tokenize(text)
            # print(tokens)
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    # Adds the words to the vocabulary.

    vocab.add_word(Constants.PAD_WORD)
    vocab.add_word(Constants.UNK_WORD)
    vocab.add_word(Constants.BOS_WORD)
    vocab.add_word(Constants.EOS_WORD)
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word in word_count.keys():
            count = word_count[word]
            vocab.word_count.append(1/count)
        else:
            vocab.word_count.append(int(0))
    return vocab

def build_vocab_tag(path, threshold):
    counter = Counter()
    for i, filename in enumerate(os.listdir(path)):
        if i % 500 == 0:
            print(f"caption: {i}")
        fp = os.path.join(path, filename)
        lines = open(fp, 'r').readlines()
        for line in lines:
            text = line.strip()
            tags= text.replace(":", " ").lower().split(' ')
            # print(tokens)
            counter.update(tags)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    # Adds the words to the vocabulary.

    vocab.add_word(Constants.PAD_WORD)
    vocab.add_word(Constants.UNK_WORD)
    vocab.add_word(Constants.BOS_WORD)
    vocab.add_word(Constants.EOS_WORD)
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word in word_count.keys():
            count = word_count[word]
            vocab.word_count.append(1/count)
        else:
            vocab.word_count.append(int(1))
    return vocab

def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    cap_path = os.path.join(args.data_path, 'descriptions_train/')
    # print(cap_path)
    vocab_cap = build_vocab_cap(path=cap_path, threshold=args.threshold)
    vocab_path = os.path.join(args.vocab_dir, f'vocab_cap.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_cap, f)

    print("Total vocabulary size: %d" %len(vocab_cap))
    print(vocab_cap.word2idx)
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)

    # Tags Tok
    tag_path = os.path.join(args.data_path, 'tags_train/')
    vocab_tag = build_vocab_tag(path=tag_path, threshold=args.threshold)
    vocab_tag_path = os.path.join(args.vocab_dir, f'vocab_tag.pkl')
    with open(vocab_tag_path, 'wb') as f:
        pickle.dump(vocab_tag, f)

    print("Total tag vocabulary size: %d" %len(vocab_tag))
    print(vocab_tag.word2idx)
    print("Saved the tag vocabulary wrapper to '%s'" %vocab_tag_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dir', type=str, default='./',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--data_path', type=str, default='../data',
                         help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')

    args = parser.parse_args()
    main(args)
