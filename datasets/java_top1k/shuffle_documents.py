from io import StringIO
import argparse
import random
import sys

from tqdm import tqdm

from .args import add_dict_options, opt, OptionEnum


def main():
    ARGS = [
        OptionEnum.SEED.value.default(0),
        opt('--output-files', type=str, nargs='+', required=True)
    ]

    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    random.seed(args.seed)

    files = [open(x, 'w') for x in args.output_files]
    document = []
    curr_file = random.choice(files)
    for line in tqdm(sys.stdin):
        line = line.strip()
        document.append(line)
        if line == '<><END OF ARTICLE><>':
            curr_file = random.choice(files)
            if len(document) > 0:
                print('\n'.join(document), file=curr_file)
                document = []


if __name__ == '__main__':
    main()