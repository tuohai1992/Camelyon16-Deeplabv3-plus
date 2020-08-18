#!/bin/bash
#SBATCH -N 1
#SBATCH -p fat_soil_shared
#SBATCH -c 20
#SBATCH -t 08:00:00

python /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_3_patch/CAMELYON16_PREPROCESSING16_preprocess.py