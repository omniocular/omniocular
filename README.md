<p align="center">
<img src="https://github.com/achyudh/omniocular/blob/master/docs/omniocular_logo.png" width="480">
</p>

Omniocular is a framework for building deep learning models on code, implemented in PyTorch by the Data Systems Group at the University of Waterloo. Various modules in Omniocular are heavily inspired by (and are compatible with) [Hedwig](https://github.com/castorini/hedwig), a framework for dcoument classification.

## Models
### Predictions over a single sequence of tokens
+ [Reg-CNN](models/diff_token/reg_cnn): Convolutional networks with regularization
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for token sequence classification
+ [HR-CNN](models/han/): Hierarchical Convolutional Networks with regularization

### Predictions over a paired sequence of tokens
+ [Reg-CNN](models/diff_token/reg_cnn): Convolutional networks with regularization
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for token sequence classification
+ [HR-CNN](models/han/): Hierarchical Convolutional Networks with regularization

### Embeddings for code
+ [Token2vec](embeddings/token2vec): Word2vec-based embeddings for programming language tokens
+ [Code2vec](embeddings/code2vec): Distributed representations for code from collections of AST paths

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
    ├── embeddings
    └── datasets
```

After cloning the omniocular-data repo, check if you have a text file containing the embeddings for Java:

```bash
cd omniocular-data/embeddings/
ls java1k_size300_min10.bin.txt 
```
