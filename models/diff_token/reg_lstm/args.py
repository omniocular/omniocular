import os

import models.diff_token.args


def get_args():
    parser = models.diff_token.args.get_args()

    parser.add_argument('--dataset', type=str, default='ApacheDiffToken', choices=['ApacheDiffToken', 'SpringDiffToken'])
    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--bottleneck-layer', action='store_true')
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=256)

    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--wdrop', type=float, default=0.0, help="weight drop")
    parser.add_argument('--beta-ema', type=float, default=0, help="temporal averaging")
    parser.add_argument('--embed-droprate', type=float, default=0.0, help="embedding dropout")
    parser.add_argument('--tar', type=float, default=0.0, help="temporal activation regularization")
    parser.add_argument('--ar', type=float, default=0.0, help="activation regularization")

    parser.add_argument('--word-vectors-dir', default=os.path.join(os.pardir, 'omniocular-data', 'embeddings'))
    parser.add_argument('--word-vectors-file', default='java1k_size300_min10.txt')
    parser.add_argument('--save-path', type=str, default=os.path.join('models', 'diff_token', 'reg_lstm', 'saves'))
    parser.add_argument('--resume-snapshot', type=str)
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
