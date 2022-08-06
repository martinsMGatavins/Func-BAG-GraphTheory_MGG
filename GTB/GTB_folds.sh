#!/bin/bash

declare -a ARRAY=("alff" "reho" "intrafc" "interfc"
                  "eff" "part" "clust" "bc" "str")

projectdir=/home/gatavins/brain_age
masteroutputdir=${projectdir}/outputs/10folds
if [ ! -e ${masteroutputdir} ]
then
  mkdir ${masteroutputdir}
  echo "Created output directory"
fi

tempdir=${projectdir}/code/temp
if [ ! -e ${tempdir} ]
then
  mkdir ${tempdir}
  echo "Created tempdir"
fi

module load python
for SUB in "${ARRAY[@]}"
do

category=${SUB}_10Fold
mkdir ${masteroutputdir}/${category}
exitdir=${masteroutputdir}/${category}

targetdir=${projectdir}/data/sets_for10folds
inputfile=${targetdir}/${SUB}.csv
foldfile=${targetdir}/${SUB}

cd temp
cat <<EOF > folding_${SUB}.py
#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

full_data = pd.read_csv("${inputfile}")
data = full_data['sub'].to_numpy()

# any other random state works too
kfold = KFold(n_splits=10,shuffle=True,random_state=23)

a=0
for train, test in kfold.split(data):
  train_full = full_data.iloc[train]
  test_full = full_data.iloc[test]
  train_full.to_csv("${foldfile}_train_" + str(a) + ".csv")
  test_full.to_csv("${foldfile}_test_" + str(a) + ".csv")
EOF
cd ..
python temp/folding_${SUB}.py

for ((i=0; i<10; i++))
do

testfile=${foldfile}_test_"$i".csv
trainfile=${foldfile}_train_"$i".csv

cat <<EOF > "trainingcall_$i.sh"
#!/bin/sh
#SBATCH --job-name=GTBtrainingbyFold
#SBATCH -o /scratch/gatavins/training_%x.stdout
#SBATCH -e /scratch/gatavins/training_%x.stderr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 4G
#SBATCH -t 03:00:00

module load python
python GTB_foldruns.py ${trainfile} ${testfile} ${exitdir} $i

EOF

sbatch trainingcall_$i.sh
done
echo "Done with ${SUB}"
done

exit 0
