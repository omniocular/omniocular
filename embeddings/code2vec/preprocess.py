from collections import defaultdict
import os

path_contexts_file = "astminer_output/path_contexts.csv"
tokens_file = "astminer_output/tokens.csv"
paths_file = "astminer_output/paths.csv"
split_path = "/mnt/collections/j9xin/astminerGT/testData/vulas_diff/"

def rec_dd():
    return defaultdict(rec_dd) #magic!


def read_ast_paths_label(fname):
    data = rec_dd()
    with open(fname) as f:
        for idx, line in enumerate(f):
            if idx==0:
                continue
            a = line.strip().split(',')
            sha, num, label = a[0].split('/')[-1][:-5].split('_')
            bag_paths = list(set(
                map(lambda x: '\t'.join(x.split(' ')), a[1].split(';'))
            ))
            if label=="00":
                data[sha][num]["prev"] = bag_paths
            else:
                data[sha][num]["curr"] = bag_paths #curr for current
                data[sha]["label"] = "positive" if label=="10" else "negative"
    return data
    # data[sha] = {label: "positive", 0:{"prev": bag, "curr": bag}, 
    #                                  1:{"prev": bag, "curr": bag}, ...}


def read_labels(pathname):
    """
    deprecated
    """
    data = {"train":{}, "dev":{}, "test":{}}
    for split in ["train", "dev", "test"]:
        with open(pathname+split+".tsv") as f:
            for idx, line in enumerate(f):
                a = line.strip().split('\t')
                repo = a[0]
                label = a[3]
                data[split][idx] = [repo, label]
    return data


def get_split(path_name):
    data = {"train":set(), "dev":set(), "test":set()}

    for split in ["train", "dev", "test"]:
        for f in os.listdir(path_name+split):
            data[split].add(f.split('_')[0])

    return data



def generate_corpus(ast_paths, splits):
    count = {"train": 0, "dev": 0, "test": 0}
    for split in ["train", "dev", "test"]:
        with open(split+".txt", 'w') as fout:
            for sha in ast_paths:
                if sha in splits[split]:
                    prev_path_set = set()
                    curr_path_set = set()
                    for key in ast_paths[sha]:
                        # key: which file in the commit
                        if key=="label":
                            continue
                        for path in ast_paths[sha][key]["prev"]:
                            if path=='':
                                continue
                            prev_path_set.add(path)
                        for path in ast_paths[sha][key]["curr"]:
                            if path=='':
                                continue
                            curr_path_set.add(path)
                    prev_paths = prev_path_set.difference(curr_path_set)
                    curr_paths = curr_path_set.difference(prev_path_set)
                    if len(prev_paths)>0 and len(curr_paths)>0:
                        print('#'+str(count[split]), file=fout)
                        print('label:'+ast_paths[sha]["label"], file=fout)
                        print('class:'+"/unk.java", file=fout)
                        print('prev_paths:', file=fout)
                        print('\n'.join(prev_paths), file=fout)
                        print('curr_paths:', file=fout)
                        print('\n'.join(curr_paths), file=fout)
                        print('vars:', file=fout)
                        print('UNK\tUNK', file=fout)
                        print(file=fout)
                    count[split] += 1


def generate_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            if idx==0:
                continue
            flag = line.find(',')
            a = [line[:flag], line[flag+1:-1]]
            try:
                vocab[int(a[0])] = a[1]
            except ValueError: #this does happen, there exist '\n' in a[1]
                pass

    foutname = filename.replace("csv", "txt").split('/')[1]
    with open(foutname, 'w') as fout:
        for i_1 in range(len(vocab)):
            i = i_1 + 1
            print("{}\t{}".format(i, vocab[i]), file=fout)


splits = get_split(split_path)
ast_paths = read_ast_paths_label(path_contexts_file)
generate_corpus(ast_paths, splits)
generate_vocab(tokens_file)
generate_vocab(paths_file)
