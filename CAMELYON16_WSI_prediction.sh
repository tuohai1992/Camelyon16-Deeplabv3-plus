#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_titanrtx
#SBATCH -t 2-00:00:00

python /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_3_patch/CAMELYON16_WSI_prediction.py