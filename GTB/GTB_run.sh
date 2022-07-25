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

projectdir=/home/gatavins/brain_age
inputdir=${projectdir}/data/train_test_tempdata

declare -a ARRAY=("alff" "reho" "intrafc" "interfc" "modul"
                  "centr" "eff" "part" "clust" "bc" "str")
i=`expr ${SLURM_ARRAY_TASK_ID} - 1`
SUB=${ARRAY[$i]}

category=${SUB}_firstround
mkdir ${projectdir}/outputs/${category}
outputdir=${projectdir}/outputs/${category}

module load python
python GTB_firstround_versionA.py \
  "${inputdir}/${SUB}_train.csv" \
  "${inputdir}/${SUB}_test.csv" \
  "${category}"

exit 0
