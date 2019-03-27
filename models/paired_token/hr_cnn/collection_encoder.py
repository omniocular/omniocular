from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.paired_token.hr_cnn.file_encoder import FileEncoder


class CollectionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_channel = config.output_channel
        self.mode = config.mode
        self.batchnorm = config.batchnorm
        self.beta_ema = config.beta_ema
        self.dynamic_pool = config.dynamic_pool
        self.dynamic_pool_length = config.dynamic_pool_length
        self.has_bottleneck = config.bottleneck_layer
        self.bottleneck_units = config.bottleneck_units
        self.sentence_encoder = FileEncoder(config)
        self.filter_widths = 3

        input_channel = 3
        input_channel_dim = config.file_channel * self.dynamic_pool_length \
            if self.dynamic_pool \
            else config.file_channel

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (3, input_channel_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (5, input_channel_dim), padding=(4, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (7, input_channel_dim), padding=(6, 0))

        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.output_channel)
            self.batchnorm2 = nn.BatchNorm2d(self.output_channel)
            self.batchnorm3 = nn.BatchNorm2d(self.output_channel)

        self.dropout = nn.Dropout(config.dropout)

        if self.dynamic_pool:
            self.dynamic_pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Dynamic pooling

    def forward(self, x, **kwargs):
        x = x.permute(1, 0, 2)  # (sentences, batch size, words)
        num_sentences = x.size()[0]
        x_encoded = list()
        for i in range(num_sentences):
            x_encoded.append(self.sentence_encoder(x[i, :, :]))

        x = torch.stack(x_encoded)  # (sentences, channels, batch size, words)
        x = x.permute(2, 1, 0, 3)  # (batch size, channels, sentences, words)

        if self.batchnorm:
            x = [F.relu(self.batchnorm1(self.conv1(x))).squeeze(3),
                 F.relu(self.batchnorm2(self.conv2(x))).squeeze(3),
                 F.relu(self.batchnorm3(self.conv3(x))).squeeze(3)]
        else:
            x = [F.relu(self.conv1(x)).squeeze(3),
                 F.relu(self.conv2(x)).squeeze(3),
                 F.relu(self.conv3(x)).squeeze(3)]

        if self.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch, channel_output) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)
            x = x.view(-1, self.filter_widths * self.output_channel * self.dynamic_pool_length)
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, channel_output, ~=sent_len) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)

        return x
