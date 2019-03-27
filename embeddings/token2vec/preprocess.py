import ast
import numbers

import astor
import javalang
import timeout_decorator
from nltk import RegexpTokenizer


def is_numeric(obj):
    """
    Check if the given object represents a number
    :param obj: input object
    :return: True if obj is a number, else False
    """
    if isinstance(obj, numbers.Number):
        return True
    elif isinstance(obj, str):
        try:
            nodes = list(ast.walk(ast.parse(obj)))[1:]
        except SyntaxError:
            return False
        if not isinstance(nodes[0], ast.Expr):
            return False
        if not isinstance(nodes[-1], ast.Num):
            return False
        nodes = nodes[1:-1]
        for i in range(len(nodes)):
            if i % 2 == 0:
                if not isinstance(nodes[i], ast.UnaryOp):
                    return False
            else:
                if not isinstance(nodes[i], (ast.USub, ast.UAdd)):
                    return False
        return True
    else:
        return False


def is_ascii(string):
    return all(ord(c) < 128 for c in string)


@timeout_decorator.timeout(10)
def tokenize_code(blob, language='java'):
    """

    :param blob:
    :param language:
    :return:
    """
    if language == 'python':
        parsed_code = astor.to_source(ast.parse(blob))
        tokenized_code = [x for x in RegexpTokenizer(r'\w+').tokenize(parsed_code) if not is_numeric(x)]
        return tokenized_code

    elif language == 'java':
        return [''.join(x.value.split()) for x in javalang.tokenizer.tokenize(blob)
                if len(x.value.strip()) < 64 and is_ascii(x.value)]

    else:
        raise Exception("Unsupported language")
