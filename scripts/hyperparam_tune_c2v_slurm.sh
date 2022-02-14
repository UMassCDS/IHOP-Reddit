#!/bin/bash
#SBATCH -c 20
#SBATCH -p cpu-long
#SBATCH --mem=8000
#SBATCH -t 10-00:00
#SBATCH -o c2v-hyperparam-slurm-RC_2021-06_5percentTopUsersExcluded_01282022_models2.out

#DATA_ROOT="data/community2vec/sample_data"
#DATA_ROOT="data/community2vecs3/RC_2021-06_0percentTopUsersExcluded_01282022"
#DATA_ROOT="data/community2vecs3/RC_2021-06_2percentTopUsersExcluded_01282022"
DATA_ROOT="data/community2vecs3/RC_2021-06_5percentTopUsersExcluded_01282022"
#DATA_ROOT="data/community2vecs3/RC_2021-06_10percentTopUsersExcluded_01282022"
CONTEXTS="${DATA_ROOT}/user_contexts"
VOCAB="${DATA_ROOT}/subreddit_counts.csv"
MODEL_DIR="${DATA_ROOT}/models2"

module load miniconda/4.8.3
module load java/11.0.2

# You only have to install once, uncomment next 4 lines if you haven't done so
#conda create --yes --name ihop python=3.8
#conda activate ihop
#pip install -r requirements.txt
#pip install .

python -m ihop.community2vec --contexts $CONTEXTS --vocab_csv $VOCAB --output_dir $MODEL_DIR --param_grid '{"alpha": [0.08,0.1], "vector_size":[150], "sample":[0.001, 0.005, 0.05], "negative":[10,20,40]}' --workers 20

module purge