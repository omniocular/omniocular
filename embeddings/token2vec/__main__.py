import os

from embeddings.token2vec import model
from embeddings.token2vec.args import get_args


def load_dataset(input_path, filter_extension=None):
    """
    :param input_path:
    :param filter_extension:
    :return:
    """
    code_repository = list()
    for root, dirs, files in os.walk(input_path):
        for file_path in files:
            if file_path.endswith(filter_extension):
                try:
                    with open(os.path.join(root, file_path), 'r', encoding='utf-8') as code_file:
                        code_repository.append(code_file.read())

                except UnicodeDecodeError:
                    print('UnicodeDecodeError:', os.path.join(root, file_path))

                except FileNotFoundError:
                    print('FileNotFoundError:', os.path.join(root, file_path))

    return code_repository


if __name__ == '__main__':
    args = get_args()

    dataset_ext_map = {'Python1K': '.py', 'Java1K': '.java'}
    code_repository = load_dataset(os.path.join(args.data_dir, args.dataset.lower()),
                                   filter_extension=dataset_ext_map[args.dataset])
    print('Size of code repository:', len(code_repository))

    trained_model = model.train(code_repository, args)
    trained_model.wv.save_word2vec_format(os.path.join(args.save_dir, '%s_size%s_min%s.bin' % (args.dataset.lower(), args.embed_dim, args.min_count)), binary=True)
    trained_model.wv.save_word2vec_format(os.path.join(args.save_dir, '%s_size%s_min%s.txt' % (args.dataset.lower(), args.embed_dim, args.min_count)), binary=False)
