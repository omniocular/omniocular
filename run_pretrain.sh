output="bert_pretrain_output_1M"
checkpoint=${output}/checkpoint
mkdir -p $output
rm -f ${checkpoint}/*
python -m embeddings.bert --on-memory --log-dir $output --data-path ../omniocular-data/datasets/github_corpus/java_encode_1M.txt --model bert-base-cased --output-dir $checkpoint
