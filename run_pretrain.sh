output="bert_pretrain_output"
checkpoint=${output}/checkpoint
mkdir -p $output
rm -f ${checkpoint}/*
python -u -m models.bert_pretrain --train_corpus ../omniocular-data/datasets/vulas_diff_token/1M.tsv --bert_model bert-base-uncased --output_dir $checkpoint --do_train --local_rank -1
