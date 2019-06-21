output="bert_pretrain_output_1M"
checkpoint=${output}/checkpoint
mkdir -p $output
rm -f ${checkpoint}/*
python -m embeddings.bert --on-memory --log-dir $output --data-path ../omniocular-data/datasets/github_corpus/1M.tsv --model bert-base-uncased --output-dir $checkpoint
