import re

import javalang


def string(string, max_length=5000):
    """
    Performs tokenization and string cleaning
    """
    string = re.sub(r'[^A-Za-z0-9]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    tokenized_string = string.lower().strip().split()
    return tokenized_string[:min(max_length, len(tokenized_string))]


def java(string, max_length=5000):
    """

    :param string:
    :param max_length:
    :return:
    """
    try:
        tokenized_string = [x.value for x in javalang.tokenizer.tokenize(string)]
        return tokenized_string[:min(max_length, len(tokenized_string))]

    except Exception as ex:
        print(type(ex).__name__ + ":", string[:100])
        return string(string, max_length)
