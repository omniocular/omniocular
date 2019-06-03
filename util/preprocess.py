import json


def split_string(string, max_length=2000):
    split_val = string.split()
    return split_val[:min(max_length, len(split_val))]


def split_json_string(string, max_length=8000):
    split_val = ' '.join(json.loads(string)).split()
    return split_val[:min(max_length, len(split_val))]


def split_json(string, max_length=10):
    split_val = json.loads(string)
    return split_val[:min(max_length, len(split_val))]


def remove_field(*args, **kwargs):
    return 0


def process_labels(string):
    """
    Returns the label string as a list of integers
    :param string:
    :return:
    """
    return [float(x) for x in string]