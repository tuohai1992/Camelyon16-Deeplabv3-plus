# Camelyon16-Deeplabv3-plus
Deeplabv3+ training and prediction pipeline for camelyon16 dataset

## Camelyon16 data preprocessing description (TODO):

## Training :

### Options for model training (can be set up in train.sh):

optional arguments:

```
--img_size         : Image size to use (default:1024)
--epochs           : number of epochs for training (default:20)
--batch_size       : Batch size to use (default:2)
--num_steps        : Number of steps for training (default:50000)
--neg_pos_ratio    : Ratio of negative to positive sample in data set for training
--horovod          : Distributed training via horovod'(default:True)
--flip             : Flip training image horizontally (left to right) (default:True)
--augment          : Implement color argumentation (default:True)
--fp16_allreduce   : Reduce to FP16 precision for gradient all reduce (default:True)
--cuda             : Use CUDA or not (default:True)
--model_path.      : path of deeplabv3 model (default:None)
--patch_path       : path of extracted patches (default:None)
--log_dir          : Folder of where the logs are saved (default:None)
--weight_path      : Folder of where the weights are saved (default:None)


```
### Running on LISA GPU

Start training with CAMELYON16 dataset on LISA GPU with patch size 1024x1024, on two titanrtx GPU nodes

```
mpirun -oversubscribe -map-by ppr:2:socket -np 8 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python /home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/github_camelyon/train4.py \
    --img_size  1024 \
    --epochs    20 \
    --batch_size 2 \
    --num_steps  50000 \
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
```
### Requirements(training)
* `python = 3.7.4`
* `tensorflow-gpu = 2.0.0`
* `horovod = 0.18.2`
* `cuda = 10.0`
* `sklearn`


## Camelyon16 data postprocessing description (TODO):
