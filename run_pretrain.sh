output="bert_pretrain_output"
checkpoint=${output}/checkpoint
mkdir -p $output
rm -f ${checkpoint}/*
python -m embeddings.bert --on-memory --data-path ../omniocular-data/datasets/vulas_diff_token/1M.tsv --model bert-base-uncased --output-dir $checkpoint
