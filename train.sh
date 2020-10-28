#!/bin/bash
#SBATCH -N 2
#SBATCH -p gpu_titanrtx
#SBATCH -t 3-00:00:00
#SBATCH -n 8
#SBATCH -c 6

mpirun -oversubscribe -map-by ppr:2:socket -np 8 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/github_camelyon16/train.py \
    --img_size  1024 \
    --epochs    50 \
    --batch_size 2 \
    --num_steps  2000 \
    --neg_pos_ratio 3 \
    --horovod  True \
    --flip     True \
    --augment  True \
    --fp16_allreduce True \
    --cuda     True \
    --model_path  ./keras-deeplab-v3-plus-master \
    --patch_path  /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_2_patch/patch_1024_came16_level2 \
    --log_dir     /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_3_patch/tmp/train_data \
    --weight_path /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_1_patch/tmp_cam16 \
