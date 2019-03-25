# Adapted from Isao Sonobe's code: https://github.com/sonoisa/code2vec

import re
import random
import logging

import torch
from torch.utils.data import Dataset


logger = logging.getLogger()

""" from dataset.py """


class CodeDataset(Dataset):
    """dataset for training/test"""

    def __init__(self, ids, starts, paths, ends, labels, transform=None):
        self.ids = ids
        self.starts = starts
        self.paths = paths
        self.ends = ends
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        item = {
            'id': self.ids[index],
            'starts': self.starts[index],
            'paths': self.paths[index],
            'ends': self.ends[index],
            'label': self.labels[index]
        }
        if self.transform:
            item = self.transform(item)
        return item


class CodeData(object):
    """data corresponding to one method"""

    def __init__(self):
        self.id = None
        self.label = None
        self.normalized_label = None
        self.path_contexts = []
        self.source = None
        self.aliases = {}


class Vocab(object):
    """vocabulary (terminal symbols or path names or label(method names))"""

    REDUNDANT_SYMBOL_CHARS = re.compile(r"[_0-9]+")
    METHOD_SUBTOKEN_SEPARATOR = re.compile(
        r"([a-z]+)([A-Z][a-z]+)|([A-Z][a-z]+)")

    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.itosubtokens = {}
        self.freq = {}
        self.size = 0

    def append(self, name, index=None, subtokens=None):
        self.size += 1
        if name not in self.stoi:
            if index is None:
                index = len(self.stoi)
            if self.freq.get(index) is None:
                self.freq[index] = 0
            self.stoi[name] = index
            self.itos[index] = name
            if subtokens is not None:
                self.itosubtokens[index] = subtokens
            self.freq[index] += 1

    def get_freq_list(self):
        freq = self.freq
        freq_list = [0] * self.len()
        for i in range(self.len()):
            freq_list[i] = freq[i]
        return freq_list

    def duplicate_len(self):
        return self.size

    def len(self):
        return len(self.stoi)

    @staticmethod
    def normalize_method_name(method_name):
        return Vocab.REDUNDANT_SYMBOL_CHARS.sub("", method_name)

    @staticmethod
    def get_method_subtokens(method_name):
        ans = [
            x.lower()
            for x in Vocab.METHOD_SUBTOKEN_SEPARATOR.split(method_name)
            if x is not None and x != ''
        ]
        return ans

""" from dataset_reader.py """


class VocabReader(object):
    """read vocabulary file"""

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        vocab = Vocab()
        with open(self.filename, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip(' \r\n\t').split('\t')
                index = int(data[0])
                if len(data) > 1:
                    name = data[1]
                else:
                    name = ""
                vocab.append(name, index)
        return vocab


class DatasetReader(object):
    """read dataset file"""

    def __init__(self, corpus_path, path_index_path, terminal_index_path):
        self.path_vocab = VocabReader(path_index_path).read()
        logger.info('path vocab size: {0}'.format(self.path_vocab.duplicate_len()))

        self.terminal_vocab = VocabReader(terminal_index_path).read()
        logger.info('terminal vocab size: {0}'.format(
            self.terminal_vocab.duplicate_len()))

        self.label_vocab = Vocab()
        self.items = {"train": [], "dev": [], "test": []}
        self.load(corpus_path)

        logger.info('label vocab size: {0}'.format(self.label_vocab.len()))
        logger.info('train corpus: {0}'.format(len(self.items["train"])))

    def load(self, corpus_path):
        for split in ["train", "dev", "test"]:
            with open(corpus_path+split+".txt", mode="r", encoding="utf-8") as f:
                code_data = None
                path_contexts_append = None
                parse_mode = 0
                for line in f.readlines():
                    line = line.strip(' \r\n\t')

                    if line == '':
                        if code_data is not None:
                            self.items[split].append(code_data)
                            code_data = None
                        continue

                    if code_data is None:
                        code_data = CodeData()
                        path_contexts_append = code_data.path_contexts.append

                    if line.startswith('#'):
                        code_data.id = int(line[1:])
                    elif line.startswith('label:'):
                        label = line[6:]
                        code_data.label = label
                        normalized_label = Vocab.normalize_method_name(label)
                        subtokens = Vocab.get_method_subtokens(normalized_label)
                        normalized_lower_label = normalized_label.lower()
                        code_data.normalized_label = normalized_lower_label
                        self.label_vocab.append(
                            normalized_lower_label, subtokens=subtokens)
                    elif line.startswith('class:'):
                        code_data.source = line[6:]
                    elif line.startswith('paths:'):
                        parse_mode = 1
                    elif line.startswith('vars:'):
                        parse_mode = 2
                    elif line.startswith('doc:'):
                        doc = line[4:]
                    elif parse_mode == 1:  # paths
                        path_context = line.split('\t')
                        path_contexts_append((int(path_context[0]), int(
                            path_context[1]), int(path_context[2])))
                    elif parse_mode == 2:  # vars
                        alias = line.split('\t')
                        code_data.aliases[alias[1]] = alias[0]

                if code_data is not None:
                    self.items[split].append(code_data)

""" from dataset_bulder.py """


class DatasetBuilder(object):
    """transform dataset for training and test"""

    def __init__(self, reader, option, split_ratio=0.2):
        self.reader = reader
        self.option = option

        # random.shuffle(reader.items)
        train_items = reader.items["train"]
        dev_items = reader.items["dev"]
        test_items = reader.items["test"]
        train_count = len(train_items)
        dev_count = len(dev_items)
        test_count = len(test_items)
        logger.info('train dataset size: {0}'.format(len(train_items)))
        logger.info('test dataset size: {0}'.format(len(test_items)))

        self.train_items = train_items
        self.dev_items = dev_items
        self.test_items = test_items

    def refresh_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(
            self.reader, self.train_items, self.option.max_path_length)
        self.train_dataset = CodeDataset(
            inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(
            self.reader, self.dev_items, self.option.max_path_length)
        self.dev_dataset = CodeDataset(
            inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(
            self.reader, self.test_items, self.option.max_path_length)
        self.test_dataset = CodeDataset(
            inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)


    # def refresh_test_dataset(self):
    #     """refresh test dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
    #     inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(
    #         self.reader, self.test_items, self.option.max_path_length)
    #     self.test_dataset = CodeDataset(
    #         inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

    def build_data(self, reader, items, max_path_length):
        inputs_id = []
        inputs_starts = []
        inputs_paths = []
        inputs_ends = []
        inputs_label = []
        label_vocab_stoi = reader.label_vocab.stoi
        for item in items:
            inputs_id.append(item.id)
            label_index = label_vocab_stoi[item.normalized_label]
            inputs_label.append(label_index)
            starts = []
            paths = []
            ends = []

            # random.shuffle(item.path_contexts)
            for start, path, end in item.path_contexts[:max_path_length]:
                starts.append(start)
                paths.append(path)
                ends.append(end)
            starts = self.pad_inputs(starts, max_path_length)
            paths = self.pad_inputs(paths, max_path_length)
            ends = self.pad_inputs(ends, max_path_length)
            inputs_starts.append(starts)
            inputs_paths.append(paths)
            inputs_ends.append(ends)
        inputs_starts = torch.tensor(inputs_starts, dtype=torch.long)
        inputs_paths = torch.tensor(inputs_paths, dtype=torch.long)
        inputs_ends = torch.tensor(inputs_ends, dtype=torch.long)
        inputs_label = torch.tensor(inputs_label, dtype=torch.long)
        return inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label

    def pad_inputs(self, data, length, pad_value=0):
        """pad values"""

        assert len(data) <= length

        count = length - len(data)
        data.extend([pad_value] * count)
        return data
