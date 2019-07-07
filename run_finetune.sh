#ptm="bert-base-cased"
ptm="bert_pretrain_output_10M/checkpoint/e2-s9.bin"
python -m models.diff_token.bert --model bert-base-cased --batch-size 4 --max-file 8 --max-line 64 --dropout 0.5 --pretrained-model $ptm --freeze
