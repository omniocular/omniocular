import csv
import os
import sys

import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from util.preprocess import split_string, remove_field, process_labels, split_json, split_json_string

# Increase the upper limit on parsed fields
csv.field_size_limit(sys.maxsize)


class SpringDiffToken(TabularDataset):
    NAME = 'SpringDiffToken'
    NUM_CLASSES = 3
    IS_MULTILABEL = False

    REPO_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=remove_field)
    SHA_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=remove_field)
    CODE_FIELD = Field(batch_first=True, tokenize=split_json_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.code)

    @classmethod
    def splits(cls, path, train=os.path.join('spring_diff_token', 'train.tsv'),
               validation=os.path.join('spring_diff_token', 'dev.tsv'),
               test=os.path.join('spring_diff_token', 'test.tsv'), **kwargs):
        return super(SpringDiffToken, cls).splits(
            path, train=train, validation=validation, test=test, format='tsv',
            fields=[('repo', cls.REPO_FIELD), ('sha', cls.SHA_FIELD), ('code', cls.CODE_FIELD), ('label', cls.LABEL_FIELD)]
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
        cls.CODE_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class SpringDiffTokenHierarchical(SpringDiffToken):
    NESTING_FIELD = Field(batch_first=True, tokenize=split_string)
    CODE_FIELD = NestedField(NESTING_FIELD, tokenize=split_json)
