from typing import Tuple, List
from io import StringIO
import argparse
import random
import sys

from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from .args import OptionEnum, add_dict_options, opt


def generate_sentence(sentence, vocab) -> Tuple[List[str], List[str], List[str]]:
    gen_sentence = []
    train_mask = []
    truth_words = []
    for word in sentence:
        roll = random.random()
        if roll > 0.15:
            gen_sentence.append(word)
            train_mask.append('0')
            continue
        train_mask.append('1')
        truth_words.append(word)
        roll /= 0.15
        if roll < 0.8:
            gen_sentence.append('[MASK]')
        elif roll < 0.9:
            gen_sentence.append(word)
        else:
            gen_sentence.append(random.choice(vocab))
    return truth_words, gen_sentence, train_mask


def generate_data(documents, vocab, dupe_factor):
    all_sents = []
    list(map(all_sents.extend, documents))
    print('\t'.join(('truth1', 'truth2', 'gen_sent1', 
                     'gen_sent2', 'train_mask1', 'train_mask2', 'label')))

    count = 0
    total = 1000000 #10^6
    iter = tqdm(documents, total=total)
    for document in iter:
        for _ in range(dupe_factor):
            for idx in range(len(document)):
                sentence = document[idx]
                truth1, gen_sent1, train_mask1 = generate_sentence(sentence, vocab)
                roll = random.random()
                if roll < 0.5 and idx != len(document) - 1:
                    lbl = 'IsNext'
                    sent2 = document[idx + 1]
                else:
                    lbl = 'NotNext'
                    sent2 = random.choice(all_sents)
                truth2, gen_sent2, train_mask2 = generate_sentence(sent2, vocab)
                print(' '.join(truth1), end='\t')
                print(' '.join(truth2), end='\t')
                print(' '.join(gen_sent1), end='\t')
                print(' '.join(gen_sent2), end='\t')
                print(' '.join(train_mask1), end='\t')
                print(' '.join(train_mask2), end='\t')
                print(lbl)
                count += 1
                if count>=total:
                    iter.close()
                    return


def main():
    parser = argparse.ArgumentParser(description='Generate pretraining data following BERT\'s technique.',
                                     epilog='Usage:\ncat data | python -m oscar.run.generate_bert_pretraining_data'
                                            ' --bert-model /path/to/bert > generated.tsv')
    add_dict_options(parser, [
        OptionEnum.BERT_MODEL.value.required(False).default('bert-large-uncased'),
        OptionEnum.SEED.value,
        OptionEnum.NO_BERT_TOKENIZE.value,
        opt('--lines', type=int),
        opt('--print-tokenized', action='store_true'),
        opt('--generate-data', action='store_true'),
        opt('--dupe-factor', type=int, default=10, help='The replication factor.')
    ])
    args = parser.parse_args()
    random.seed(args.seed)
    if args.do_bert_tokenize:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    else:
        tokenizer = str.split

    documents = []
    document = []

    for line in tqdm(sys.stdin, total=args.lines):
        line = line.strip()
        if line.startswith('='): continue
        is_new_doc = line.startswith('<><END OF ARTICLE><>')
        line = line.strip()
        line = (line,) if is_new_doc else tokenizer(line)
        if args.print_tokenized: print(' '.join(line))
        if is_new_doc:
            if len(document) > 0: documents.append(document)
            document = []
        elif not args.print_tokenized:
            document.append(line)
    if len(document) > 0:
        documents.append(document)
    if args.generate_data:
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        generate_data(documents, list(bert_tokenizer.vocab.keys()), args.dupe_factor)


if __name__ == '__main__':
    main()
