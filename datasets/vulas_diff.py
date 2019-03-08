import csv
import os
import random
import re
import sys

import numpy as np
import torch
from nltk import tokenize
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

# Increase the upper limit on parsed fields
csv.field_size_limit(sys.maxsize)


def clean_string(string, sentence_droprate=0, max_length=5000):
    """
    Performs tokenization and string cleaning
    """
    if sentence_droprate > 0:
        lines = [x for x in tokenize.sent_tokenize(string) if len(x) > 1]
        lines_drop = [x for x in lines if random.randint(0, 100) > 100 * sentence_droprate]
        string = ' '.join(lines_drop if len(lines_drop) > 0 else lines)

    string = re.sub(r'[^A-Za-z0-9]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    tokenized_string = string.lower().strip().split()
    return tokenized_string[:min(max_length, len(tokenized_string))]


def split_sents(string, max_length=40):
    tokenized_string = [x for x in tokenize.sent_tokenize(string) if len(x) > 1]
    return tokenized_string[:min(max_length, len(tokenized_string))]


def char_quantize(string, max_length=1000):
    identity = np.identity(len(VulasDiffCharQuantized.ALPHABET))
    quantized_string = np.array([identity[VulasDiffCharQuantized.ALPHABET[char]]
                                 for char in list(string.lower()) if
                                 char in VulasDiffCharQuantized.ALPHABET], dtype=np.float32)

    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((
            max_length - len(quantized_string),
            len(VulasDiffCharQuantized.ALPHABET)),
            dtype=np.float32)))


def remove_field(string):
    return 0


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    return [float(x) for x in string]


class VulasDiff(TabularDataset):
    NAME = 'VulasDiff'
    NUM_CLASSES = 2
    REPO_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=remove_field)
    SHA_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=remove_field)
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('vulas_diffs', 'train.tsv'),
               validation=os.path.join('vulas_diffs', 'dev.tsv'),
               test=os.path.join('vulas_diffs', 'test.tsv'), **kwargs):
        return super(VulasDiff, cls).splits(
            path, train=train, validation=validation, test=test, format='tsv',
            fields=[('repo', cls.REPO_FIELD), ('sha', cls.SHA_FIELD), ('text', cls.TEXT_FIELD), ('label', cls.LABEL_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class VulasDiffCharQuantized(VulasDiff):
    ALPHABET = dict(map(lambda t: (t[1], t[0]),
                        enumerate(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))
    TEXT_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=char_quantize)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param batch_size: batch size
        :param device: GPU device
        :return:
        """
        train, val, test = cls.splits(path)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     device=device)


class VulasDiffHierarchical(VulasDiff):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)
