#!/bin/bash
# Export data to c2v format.

# TODO Fill in these variables.
input_compressed="data/comments/RC_2021-06.zst"
output_dir="data/community2vec/RC_2021-06"

#input_compressed="data/comments/sample_data.zst"
#output_dir="data/community2vec/sample_data"

#---------------------------------------
# Don't change script below
#---------------------------------------
# Data needs te be recompressed for spark to read due to issues with the window size and memory management, see Known Issues in Readme for more details
input_dir=$(dirname $input_compressed)
out_tmp_name=$(basename $input_compressed .zst)
temp_compressed=$input_dir/"$out_tmp_name.bz2"
echo "Recompressing $input_compressed to $temp_compressed"
unzstd --long=31 < $input_compressed | bzip2 > $temp_compressed

echo "Making output directory '$output_dir'"
mkdir -p $output_dir

python -m ihop.import_data c2v $output_dir/subreddit_counts.csv $output_dir/user_contexts $temp_compressed
