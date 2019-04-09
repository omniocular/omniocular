# Adapted from Isao Sonobe's code: https://github.com/sonoisa/code2vec

import os

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=123,
                        help="random_seed")

    parser.add_argument('--corpus_path', type=str,
                        default="../omniocular-data/datasets/vulas_diff_paths/",
                        help="corpus_path")
    parser.add_argument('--path_idx_path', type=str,
                        default="../omniocular-data/datasets/vulas_diff_paths/paths.txt",
                        help="path_idx_path")
    parser.add_argument('--terminal_idx_path', type=str,
                        default="../omniocular-data/datasets/vulas_diff_paths/tokens.txt",
                        help="terminal_idx_path")

    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch_size")
    parser.add_argument('--terminal_embed_size', type=int, default=32,
                        help="terminal_embed_size")
    parser.add_argument('--path_embed_size', type=int, default=32,
                        help="path_embed_size")
    parser.add_argument('--encode_size', type=int, default=32,
                        help="encode_size")
    parser.add_argument('--max_path_length', type=int, default=200,
                        help="max_path_length")

    parser.add_argument('--model_path', type=str, default="./output",
                        help="model_path")
    parser.add_argument('--vectors_path', type=str, default="./output/code.vec",
                        help="vectors_path")
    parser.add_argument('--test_result_path', type=str, default=None,
                        help="test_result_path")

    parser.add_argument("--max_epoch", type=int, default=40, help="max_epoch")
    parser.add_argument('--lr', type=float, default=0.001, help="lr")
    parser.add_argument('--beta_min', type=float, default=0.9, help="beta_min")
    parser.add_argument('--beta_max', type=float, default=0.999, help="beta_max")
    parser.add_argument('--weight_decay', type=float, default=1.0, help="weight_decay")

    parser.add_argument('--dropout_prob', type=float, default=0.5, help="dropout_prob")

    parser.add_argument("--no_cuda", action="store_true", help="no_cuda")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="gpu")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

    parser.add_argument("--env", type=str, default=None, help="env")
    parser.add_argument("--print_sample_cycle", type=int, default=10, help="print_sample_cycle")
    parser.add_argument("--eval_method", type=str, default="exact", help="eval_method")

    parser.add_argument("--find_hyperparams", type=bool, default=False,
                        help="find optimal hyperparameters")
    parser.add_argument("--num_trials", type=int, default=100, help="num_trials")

    args = parser.parse_args()
    return args
