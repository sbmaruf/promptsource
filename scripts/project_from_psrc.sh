set -e
# source scripts/envs/load_env.sh || true

SRC_DATA_FOLDER= path to the data folder
mkdir -p $SRC_DATA_FOLDER
mkdir -p $SRC_DATA_FOLDER/cache

python data/project_from_psrc.py \
--dataset-name-or-paths glue glue glue glue glue glue glue glue glue glue paws \
--dataset-configs mnli cola sst2 mrpc qqp stsb qnli rte wnli ax labeled_final \
--prompt-templates-configs None None None None None None None None None None None \
--cache-dir $SRC_DATA_FOLDER/cache \
--raw-output-dir $SRC_DATA_FOLDER \
--num-proc 9
