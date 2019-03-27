import gensim
from timeout_decorator.timeout_decorator import TimeoutError

from embeddings.token2vec import preprocess
from javalang.tokenizer import LexerError


def train(dataset, args):
    """
    :param input_path:
    :param language:
    :param dim:
    :param min_count:
    :return:
    """
    timeout_counter = 0
    tokenized_repository = list()

    for blob in dataset:
        try:
            tokenized_repository.append(preprocess.tokenize_code(blob))

        except TimeoutError:
            timeout_counter += 1

        except LexerError:
            print("LexerError:", ' '.join(blob[:50].split()) + "...")

        except Exception as e:
            print(type(e).__name__ + ":", ' '.join(blob[:50].split()) + "...")

        if len(tokenized_repository) % 20000 == 0:
            print("Tokenized %d of %d blobs..." % (len(tokenized_repository), len(dataset)))

    if timeout_counter:
        print("Number of timeout errors", timeout_counter)

    print("Training token2vec embeddings...")
    model = gensim.models.Word2Vec(tokenized_repository, size=args.embed_dim, workers=8, min_count=args.min_count)
    return model
