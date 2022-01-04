#!/bin/bash
# Export data to c2v format

input_dir="data/comments/RC_2021-06.zst"
output_dir="data/community2vec/2021-06"

echo "Making output directory '$output_dir'"
mkdir -p $output_dir

python -m ihop.import_data c2v $output_dir/subreddit_counts $output_dir/user_contexts $input_dir
