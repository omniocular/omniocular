import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Omniocular/Embeddings/Token2Vec")

    parser.add_argument('--dataset', type=str, default='Java1K', choices=['Java1K'])
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--min-count', type=int, default=10)

    parser.add_argument('--data-dir', default=os.path.join(os.pardir, 'omniocular-data', 'repositories'))
    parser.add_argument('--save-dir', type=str, default=os.path.join(os.pardir, 'omniocular-data', 'embeddings'))

    args = parser.parse_args()
    return args
