import os

import models.diff_token.args


def get_args():
    parser = models.diff_token.args.get_args()

    parser.add_argument('--dataset', type=str, default='VulasPairedToken', choices=['VulasPairedToken'])
    parser.add_argument('--mode', type=str, default='multichannel', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--output-channel', type=int, default=100)
    parser.add_argument('--file-channel', type=int, default=100)
    parser.add_argument('--dynamic-pool', action='store_true')
    parser.add_argument('--dynamic-pool-length', type=int, default=8)
    parser.add_argument('--bottleneck-layer', action='store_true')
    parser.add_argument('--bottleneck-units', type=int, default=50)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropblock', type=float, default=0.0)
    parser.add_argument('--dropblock-size', type=int, default=7)
    parser.add_argument('--beta-ema', type=float, default=0)
    parser.add_argument('--embed-droprate', type=float, default=0.0)
    parser.add_argument('--batchnorm', action='store_true')

    parser.add_argument('--word-vectors-dir', default=os.path.join(os.pardir, 'omniocular-data', 'embeddings'))
    parser.add_argument('--word-vectors-file', default='java1k_size300_min10.txt')
    parser.add_argument('--save-path', type=str, default=os.path.join('models', 'paired_token', 'hr_cnn', 'saves'))
    parser.add_argument('--resume-snapshot', type=str)
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
