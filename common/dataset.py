import os

import torch
import torch.nn as nn

from datasets.imdb import IMDB


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].normal_(0, 0.01)
        return cls.cache[size_tup]


class DatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device, castor_dir="./", utils_trecqa="utils/trec_eval-9.0.5/trec_eval"):
        if dataset_name == 'imdb':
            dataset_root = os.path.join(castor_dir, os.pardir, 'Castor-data', 'datasets', 'IMDB/')
            train_loader, dev_loader, test_loader = IMDB.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk)
            embedding = nn.Embedding.from_pretrained(IMDB.TEXT_FIELD.vocab.vectors)
            return IMDB, embedding, train_loader, test_loader, dev_loader
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

