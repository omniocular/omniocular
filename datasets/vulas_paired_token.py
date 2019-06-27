import csv
import os
import sys

import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from util.preprocess import remove_field, process_labels, split_json, split_string

# Increase the upper limit on parsed fields
csv.field_size_limit(sys.maxsize)


class VulasPairedToken(TabularDataset):
    NAME = 'VulasPairedToken'
    NUM_CLASSES = 2
    IS_MULTILABEL = False

    REPO_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=remove_field)
    SHA_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=remove_field)
    CODE1_FIELD = Field(batch_first=True, tokenize=split_string, include_lengths=True)
    CODE2_FIELD = Field(batch_first=True, tokenize=split_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.code1) + len(ex.code2)

    @classmethod
    def splits(cls, path, train=os.path.join('vulas_paired_token', 'train.tsv'),
               validation=os.path.join('vulas_paired_token', 'test.tsv'),
               test=os.path.join('vulas_paired_token', 'test.tsv'), **kwargs):
        return super(VulasPairedToken, cls).splits(
            path, train=train, validation=validation, test=test, format='tsv',
            fields=[('repo', cls.REPO_FIELD), ('sha', cls.SHA_FIELD),
                    ('code1', cls.CODE1_FIELD), ('code2', cls.CODE2_FIELD),
                    ('label', cls.LABEL_FIELD)]
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
        cls.CODE1_FIELD.build_vocab(train, val, test, vectors=vectors)
        cls.CODE2_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class VulasPairedTokenHierarchical(VulasPairedToken):
    NESTING1_FIELD = Field(batch_first=True, tokenize=split_string)
    CODE1_FIELD = NestedField(NESTING1_FIELD, tokenize=split_json)
    NESTING2_FIELD = Field(batch_first=True, tokenize=split_string)
    CODE2_FIELD = NestedField(NESTING2_FIELD, tokenize=split_json)
