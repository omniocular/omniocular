import os
import json

from datasets.bert_processors.abstract_processor import AbstractProcessor, InputExample


class VulasDiffTokenProcessor(AbstractProcessor):
    NAME = 'VulasDiffToken'
    NUM_CLASSES = 2
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'vulas_diff_string', 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'vulas_diff_string', 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'vulas_diff_string', 'test.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(json.loads(line[2]))
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
