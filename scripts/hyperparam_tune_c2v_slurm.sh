#!/bin/bash
#SBATCH -c 20
#SBATCH -p cpu-long
#SBATCH --mem=12000
#SBATCH -t 72:00:00
#SBATCH -o c2v-hyperparam-slurm-RC_2021-06_0percentTopUsersExcluded_01282022.out

#DATA_ROOT="data/community2vec/sample_data"
DATA_ROOT="data/community2vecs3/RC_2021-06_0percentTopUsersExcluded_01282022"
#DATA_ROOT="data/community2vecs3/RC_2021-06_2percentTopUsersExcluded_01282022"
#DATA_ROOT="data/community2vecs3/RC_2021-06_5percentTopUsersExcluded_01282022"
#DATA_ROOT="data/community2vecs3/RC_2021-06_10percentTopUsersExcluded_01282022"
CONTEXTS="${DATA_ROOT}/user_contexts"
VOCAB="${DATA_ROOT}/subreddit_counts.csv"
MODEL_DIR="${DATA_ROOT}/models"

module load miniconda/4.8.3
module load java/11.0.2

conda create --yes --name ihop python=3.8
conda activate ihop
pip install -r requirements.txt
pip install .

python -m ihop.community2vec --contexts $CONTEXTS --vocab_csv $VOCAB --output_dir $MODEL_DIR --param_grid '{"alpha": [0.05, 0.01], "vector_size":[150], "sample":[0.005, 0.05], "negative":[20,40]}' --workers 40

module purge