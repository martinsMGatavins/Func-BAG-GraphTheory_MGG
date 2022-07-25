#!/bin/sh
#SBATCH --job-name=GTBtrainingDepthTuning
#SBATCH -o /scratch/gatavins/%x_%A.%a.stdout
#SBATCH -e /scratch/gatavins/%x_%A.%a.stderr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-11
#SBATCH --mem 4G
#SBATCH -t 04:00:00
#SBATCH --mail-user=gatavins@wustl.edu
#SBATCH --mail-type=END

# set dirs for input data (already split into train and test)
# add line to parse data (or do it on local machine)
# Rscript splitting.R -> make sure to read data to train_test_tempdata (or alike)
projectdir=/home/gatavins/brain_age
inputdir=${projectdir}/data/train_test_tempdata

# list of models to train - adjust number of array jobs if you change # of models
declare -a ARRAY=("alff" "reho" "intrafc" "interfc" "modul"
                  "centr" "eff" "part" "clust" "bc" "str")
i=`expr ${SLURM_ARRAY_TASK_ID} - 1`
SUB=${ARRAY[$i]}

# makes output directory
category=${SUB}
mkdir ${projectdir}/outputs/${category}
outputdir=${projectdir}/outputs/${category}

# run command
module load python
python GTB_nested.py \
  "${inputdir}/${SUB}_train.csv" \
  "${inputdir}/${SUB}_test.csv" \
  "${category}"

exit 0
