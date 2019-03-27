import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diff_string.reg_cnn.dropblock import DropBlock1D, LinearScheduler
from models.diff_string.reg_lstm.embed_regularize import embedded_dropout


class FileEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        words_num = config.words_num
        words_dim = config.words_dim

        self.file_channel = config.file_channel
        self.mode = config.mode
        self.batchnorm = config.batchnorm
        self.embed_droprate = config.embed_droprate
        self.dynamic_pool = config.dynamic_pool
        self.dynamic_pool_length = config.dynamic_pool_length
        self.dropblock_prob = config.dropblock
        self.filter_widths = 3

        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(config.vocab, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(config.vocab, freeze=False)
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(config.vocab, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(config.vocab, freeze=False)
            input_channel = 2
        else:
            print("Unsupported Mode")
            exit()

        self.conv1 = nn.Conv2d(input_channel, self.file_channel, (3, words_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, self.file_channel, (5, words_dim), padding=(4, 0))
        self.conv3 = nn.Conv2d(input_channel, self.file_channel, (7, words_dim), padding=(6, 0))

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.file_channel)
            self.batchnorm2 = nn.BatchNorm2d(self.file_channel)
            self.batchnorm3 = nn.BatchNorm2d(self.file_channel)

        if self.dropblock_prob > 0:
            self.dropblock = LinearScheduler(
                DropBlock1D(drop_prob=self.dropblock_prob, block_size=config.dropblock_size),
                start_value=0.,
                stop_value=self.dropblock_prob,
                nr_steps=1000
            )

        if self.dynamic_pool:
            self.dynamic_pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Dynamic pooling

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            word_input = embedded_dropout(self.embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.embed(x)
            x = word_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'static':
            static_input = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
            x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'non-static':
            non_static_input = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'multichannel':
            non_static_input = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
            static_input = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)  # (batch, channel_input=2, sent_len, embed_dim)

        if self.batchnorm:
            x = [F.relu(self.batchnorm1(self.conv1(x))).squeeze(3),
                 F.relu(self.batchnorm2(self.conv2(x))).squeeze(3),
                 F.relu(self.batchnorm3(self.conv3(x))).squeeze(3)]
        else:
            x = [F.relu(self.conv1(x)).squeeze(3),
                 F.relu(self.conv2(x)).squeeze(3),
                 F.relu(self.conv3(x)).squeeze(3)]

        if self.dropblock_prob > 0:
            self.dropblock.step()
            x = [self.dropblock(conv_output) for conv_output in x]

        if self.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch, channel_output, dp) * Ks
            x = [i.view(-1, self.file_channel * self.dynamic_pool_length) for i in x]  # (batch, dp * channel_output) * Ks
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, channel_output, ~=sent_len) * Ks

        x = torch.stack(x)  # (Ks, batch, dp * channel_output)
        return x
