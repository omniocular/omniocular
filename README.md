# Omniocular
Omniocular is a framework for building deep learning models on code, implemented in PyTorch by the Data Systems Group at the University of Waterloo. 

## Models

+ [Kim CNN](models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)
+ [HAN](models/han/): Hierarchical Attention Networks [(Zichao, et al, NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for document classification [(Merity et al.)](https://arxiv.org/abs/1708.02182)
+ [XML-CNN](models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [Char-CNN](models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

Each model directory has a `README.md` with further details.

## Setting up PyTorch

Omniocular is designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4.
PyTorch recommends [Anaconda](https://www.anaconda.com/distribution/) for managing your environment.
We recommend creating a custom environment as follows:

```
$ conda create --name omniocular python=3.6
$ source activate omniocular
```

And installing PyTorch as follows:

```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```

Other Python packages we use can be installed via pip:

```
$ pip install -r requirements.txt
```

## Datasets

Download the datasets and embeddings from the 
[`omniocular-data`](https://git.uwaterloo.ca/arkeshav/omniocular-data) repository:

```bash
$ git clone https://github.com/omniocular/omniocular.git
$ git clone https://git.uwaterloo.ca/arkeshav/omniocular-data.git
```

Datasets, along with embeddings should be placed in the `omniocular-data` folder, with the following directory structure: 

```
.
├── omniocular
└── omniocular-data
```

After cloning the omniocular-data repo, you need to unzip the embeddings and run the preprocessing script:

```bash
cd omniocular-data/embeddings/word2vec 
gzip -d java1k_size300_min10.bin.bin.gz 
python bin2txt.py java1k_size300_min10.bin.bin java1k_size300_min10.bin.txt 
```

**If you are an internal Omniocular contributor using the machines in the lab, follow the instructions [here](docs/internal-instructions.md).**
