output="bert_pretrain_output_1M"
checkpoint=${output}/checkpoint
mkdir -p $output
rm -f ${checkpoint}/*
python -u -m embeddings.bert --train_corpus ../omniocular-data/datasets/vulas_diff_token/1M.tsv --bert_model bert-base-uncased --log_dir $output --output_dir $checkpoint --do_train --local_rank -1
