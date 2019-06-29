import torch
from torch import nn

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
        self.max_pool_file = nn.MaxPool1d(1000)
        self.max_pool_coll = nn.MaxPool1d(1000)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_labels)

    def forward(self, batch):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids, input_mask, segment_ids, label_ids = batch
        file_embs = []
        for i0 in range(len(input_ids)):
            line_embs = []
            for i1 in range(len(input_ids[i0])):
                line_emb = self.bert(
                    input_ids[i0][i1], input_mask[i0][i1], segment_ids[i0][i1], False)
                line_embs.append(line_emb)

            file_emb = self.max_pool_file(
                torch.stack(line_embs, 2))
            file_emb = torch.squeeze(file_emb, 2)
            file_embs.append(self.dropout(file_emb))

        coll_emb = self.max_pool_coll(
            torch.stack(file_embs, 2))
        coll_emb = torch.squeeze(coll_emb, 2)
        logits = self.classifier(coll_emb)
        return logits
