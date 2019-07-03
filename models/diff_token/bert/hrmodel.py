import torch
from torch import nn
import torch.nn.functional as F

from models.diff_token.bert.bertmodel import BertForSequenceClassification


class HRBertForSequenceClassification(nn.Module):

    def __init__(self, args, cache_dir):
        super(HRBertForSequenceClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            args.pretrained_model,
            cache_dir=cache_dir,
            num_labels=args.num_labels,
            dropout=args.dropout
        )
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_labels)

    def forward(self, batch):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids, input_mask, segment_ids, label_ids = batch  # (batch_size, files, lines, words)
        input_ids = input_ids.permute(1, 2, 0, 3)  # (files, lines, batch_size, words)
        segment_ids = segment_ids.permute(1, 2, 0, 3)
        input_mask = input_mask.permute(1, 2, 0, 3)

        file_embs = []
        for i0 in range(len(input_ids)):
            line_embs = []
            for i1 in range(len(input_ids[i0])):
                line_emb = self.bert(
                    input_ids[i0][i1], input_mask[i0][i1], segment_ids[i0][i1], False)
                line_embs.append(line_emb)

            file_emb = torch.stack(line_embs).permute(1, 2, 0)  # (batch_size, hidden_size, lines)
            file_emb = F.max_pool1d(file_emb, file_emb.size(2)).squeeze(2)  # (batch_size, hidden_size)
            file_embs.append(file_emb)

        coll_emb = torch.stack(file_embs).permute(1, 2, 0)  # (batch_size, files, hidden_size)
        coll_emb = F.max_pool1d(coll_emb, coll_emb.size(2)).squeeze(2)  # (batch_size, hidden_size)
        # coll_emb = torch.squeeze(coll_emb, 2)
        logits = self.classifier(coll_emb)
        return logits
