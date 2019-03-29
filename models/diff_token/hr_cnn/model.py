from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diff_token.hr_cnn.collection_encoder import CollectionEncoder


class HRCNN(nn.Module):

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
        self.filter_widths = 3

        config.vocab = config.dataset.CODE_FIELD.vocab.vectors
        self.collection_encoder = CollectionEncoder(config)

        self.dropout = nn.Dropout(config.dropout)

        if self.dynamic_pool:
            self.dynamic_pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Dynamic pooling
            if self.has_bottleneck:
                self.fc1 = nn.Linear(self.filter_widths * self.output_channel * self.dynamic_pool_length, self.bottleneck_units)
                self.fc2 = nn.Linear(self.bottleneck_units, config.target_class)
            else:
                self.fc1 = nn.Linear(self.filter_widths * self.output_channel * self.dynamic_pool_length, config.target_class)

        else:
            if self.has_bottleneck:
                self.fc1 = nn.Linear(self.filter_widths * self.output_channel, self.bottleneck_units)
                self.fc2 = nn.Linear(self.bottleneck_units, config.target_class)
            else:
                self.fc1 = nn.Linear(self.filter_widths * self.output_channel, config.target_class)

        if self.beta_ema > 0:
            self.avg_param = deepcopy([p.data for p in self.parameters()])
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.0

    def forward(self, x, **kwargs):
        x = self.collection_encoder(x)
        x = self.dropout(x)

        if self.has_bottleneck:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            logit = self.fc2(x)
        else:
            logit = self.fc1(x)  # (batch, target_size)
        return logit

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params