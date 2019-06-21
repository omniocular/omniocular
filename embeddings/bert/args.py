import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=3.0, type=float)
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate for adam")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument("--do-lower-case", action='store_true', help="lower case the input text")
    parser.add_argument("--max-seq-length", default=64, type=int, help="maximum input length after tokenization")
    parser.add_argument("--warmup-proportion", default=0.1, type=float,
                        help="proportion of training to perform linear learning rate warmup for")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward pass")

    parser.add_argument("--no-cuda", action='store_true')
    parser.add_argument('--fp16', action='store_true', help="use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss-scale', type=float, default=0, help="loss scaling to improve fp16 numeric stability")
    parser.add_argument("--on-memory", action='store_true', help="load the entire corpus into memory")
    parser.add_argument("--output-dir", type=str, required=True, help="directory for checkpointing models")
    parser.add_argument("--data-path", type=str, required=True, help="path to training corpus")

    args = parser.parse_args()
    return args
