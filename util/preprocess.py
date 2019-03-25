from nltk import tokenize


def split_string(string, max_length=40):
    tokenized_string = [x for x in tokenize.sent_tokenize(string) if len(x) > 1]
    return tokenized_string[:min(max_length, len(tokenized_string))]


def remove_field(*args, **kwargs):
    return 0


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    return [float(x) for x in string]