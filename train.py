from __future__ import absolute_import, division, print_function
import argparse
import os
import timeit
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras import applications
import glob
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pdb
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.python.keras.utils.data_utils import get_file

# pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('--img_size',type=int, required =True, default= 1024,help ='Image size to use' )
parser.add_argument('--epochs',type=int, required =True, default= 50,help ='number of epochs for training' )
parser.add_argument('--batch_size',type=int, required =True, default= 50,help ='Batch size to use' )
parser.add_argument('--num_steps',type=int, required =True, default= 50000,help ='Number of steps for training' )
parser.add_argument('--neg_pos_ratio',type=int, required =True, default= 1,help ='Ratio of negative to positive sample in data set for training' )
parser.add_argument('--horovod',type= str, required =True, default= 'True',help ='Distributed training via horovod' )
parser.add_argument('--flip',type= str, required =True, default= 'True',help ='Flip an image horizontally (left to right)' )
parser.add_argument('--augment',type= str, required =True, default= 'True',help ='Implement color argumentation' )
parser.add_argument('--fp16_allreduce',type= str, required =True, default= 'True',help ='Reduce to FP16 precision for gradient all reduce' )
parser.add_argument('--cuda',type= str, required =True, default= 'True',help ='Use CUDA or not' )
parser.add_argument('--model_path',type= str, required =True, default= './keras-deeplab-v3-plus-master',help ='path of deeplabv3 model' )
parser.add_argument('--patch_path',type= str, required =True, default= None ,help ='path of extracted patches' )
parser.add_argument('--log_dir',type= str, required =True, default= None,help ='Folder of where the logs are saved' )
parser.add_argument('--weight_path',type= str, required =True, default= None ,help ='Folder of where the weights are saved' )

args = parser.parse_args()

# pdb.set_trace()
sys.path.insert(0, args.model_path)

print('TensorFlow', tf.__version__)


## Set Variables ## 
IMG_SIZE                = args.img_size
EPOCHS                  = args.epochs
BATCH_SIZE              = args.batch_size
STEPS                   = args.num_steps  
RATIO                   = args.neg_pos_ratio   
H, W                    = IMG_SIZE, IMG_SIZE  
num_classes             = 2
 
if args.horovod =='True':
    HOROVOD = True
else:
    HOROVOD = False

if args.augment =='True':
    AUGMENT  = True
else:
    AUGMENT  = False

if args.flip =='True':
    FLIP  = True
else:
    FLIP  = False

if args.fp16_allreduce =='True':
    fp16_ALLREDUCE  = True
else:
    fp16_ALLREDUCE  = False

if args.cuda =='True':
    CUDA  = True
else:
    CUDA  = False


TRAIN_TUMOR_IMAGE_PATH = args.patch_path + '/image_pos'
TRAIN_TUMOR_MASK_PATH =args.patch_path +'/mask_pos'
TRAIN_NORMAL_IMAGE_PATH = args.patch_path +'/image_neg'
TRAIN_NORMAL_MASK_PATH =args.patch_path +'/mask_neg'



print("Now hvd.init")
if HOROVOD:
    hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
if CUDA:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("hvd.size() = ",hvd.size())
    print("GPU's", gpus, "with Local Rank", hvd.local_rank())
    print("GPU's", gpus, "with Rank", hvd.rank())

    if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Past hvd.init()")


image_pos_list = [ x for x in glob.glob(os.path.join(TRAIN_TUMOR_IMAGE_PATH, '*.png'))]

mask_pos_list = [ x for x in glob.glob(os.path.join(TRAIN_TUMOR_MASK_PATH, '*.png'))]

image_neg_list = [ x for x in glob.glob(os.path.join(TRAIN_NORMAL_IMAGE_PATH, '*.png'))]

mask_neg_list = [ x for x in glob.glob(os.path.join(TRAIN_NORMAL_MASK_PATH, '*.png'))]

pos_img_num = len(image_pos_list)

neg_img_num = len(image_neg_list)

neg_img_num_use= pos_img_num * RATIO

if neg_img_num_use > neg_img_num:
    image_list = image_pos_list + image_neg_list
    mask_list = mask_pos_list + mask_neg_list
    
else:

    por_use_neg= round(neg_img_num_use/neg_img_num,2)

    image_neg_use, image_neg_unuse, mask_neg_use, mask_neg_unuse = train_test_split(image_neg_list,mask_neg_list,test_size=1-por_use_neg,random_state=43)

    image_list = image_pos_list + image_neg_use
    mask_list = mask_pos_list + mask_neg_use


image_list, val_image_list, mask_list, val_mask_list = train_test_split(image_list,mask_list,test_size=0.15,random_state=42)



print('Found', len(image_list), 'training images')
print('Found', len(mask_list), 'training masks')
print('Found', len(val_image_list), 'validation images')
print('Found', len(val_mask_list), 'validation masks')


print(image_list[100:120])
print(mask_list[100:120])
print(val_image_list[100:120])
print(val_mask_list[100:120])

# pdb.set_trace()

def get_image(image_path, img_height=IMG_SIZE, img_width=IMG_SIZE, mask=False, flip=0, augment=False):

    img = tf.io.read_file(image_path)
    if not mask:
        if augment:
            aug = tf.random.uniform(shape=[1, ], minval=0, maxval=3, dtype=tf.int32)[0]
            img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
            img = tf.image.resize(images=img, size=[img_height, img_width])
            img = tf.case([
                (tf.greater(aug, 0), lambda: tf.image.random_brightness(img, max_delta=0.5))
            ], default=lambda: img)
            img = tf.case([
                (tf.greater(aug, 0), lambda: tf.image.random_saturation(img, lower=0.5, upper=1.5))
            ], default=lambda: img)
            # pdb.set_trace()
            img = tf.case([
                (tf.greater(aug, 0), lambda: tf.image.random_hue(img, max_delta=0.2))
            ], default=lambda: img)
            img = tf.case([
                (tf.greater(aug, 0), lambda: tf.image.random_contrast(img, lower=0.7, upper=1.3))
            ], default=lambda: img)
            img = tf.case([
                (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
            ], default=lambda: img)
            img = tf.clip_by_value(img, 0, 255)
            img = tf.math.divide(img,255)
            img = tf.math.subtract(tf.math.multiply(2.0,img),1.0)


        else:
            img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.int32)
            img = tf.image.resize(images=img, size=[img_height, img_width])
            # img = tf.clip_by_value(img, 0, 2)
            img = tf.math.divide(img,255)
            img = tf.math.subtract(tf.math.multiply(2.0,img),1.0)

    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
            int(img_height/4), int(img_width/4)]), dtype=tf.int32)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = tf.clip_by_value(img, 0, 1)


    return img




def load_data(image_path, mask_path, sample_weight=1, H=H, W=W, augment=False):
    if FLIP:
        flip = tf.random.uniform(
            shape=[1, ], minval=0, maxval=3, dtype=tf.int32)[0]
    else:
        flip = 0
    image, mask = get_image(image_path, flip=flip, augment=augment), get_image(
        mask_path, mask=True, flip=flip)
    return image, mask#, sample_weight



train_dataset = tf.data.Dataset.from_tensor_slices((image_list,
                                                mask_list))

train_dataset = train_dataset.shuffle(buffer_size=3000)
train_dataset = train_dataset.apply(
tf.data.experimental.map_and_batch(
    map_func=lambda x, y: load_data(image_path=x, mask_path=y, augment=AUGMENT),
    batch_size=BATCH_SIZE,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    drop_remainder=True))
# train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(train_dataset)
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,
                                                val_mask_list))

val_dataset = val_dataset.apply(
tf.data.experimental.map_and_batch(map_func=lambda x, y: load_data(image_path=x, mask_path=y, augment=AUGMENT),
                                    batch_size=BATCH_SIZE,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                    drop_remainder=True))
# val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(val_dataset)

# pdb.set_trace()

from model import Deeplabv3
model = Deeplabv3(input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=2, backbone='xception',activation='softmax')
# model.load_weights("/home/jli/examode/camelyon-master/CAMELYON16_PREPROCESSING/level_1_patch/tmp_cam16/Deeplabv3_level2_768_10per_step25000.h5", by_name= True)


if HOROVOD:
# Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if fp16_ALLREDUCE else hvd.Compression.none

    opt = tf.optimizers.Adam(0.0001 * hvd.size(),epsilon=1e-1)
# Horovod: add Horovod DistributedOptimizer.

print("Compiling model...")

model.build(input_shape=(IMG_SIZE,IMG_SIZE,3))
model.summary()




# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

size = hvd.size()

print('Preparing training...')



compute_loss     = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
compute_accuracy = tf.keras.metrics.Accuracy()
compute_miou     = tf.keras.metrics.MeanIoU(num_classes=2)
compute_auc      = tf.keras.metrics.AUC()



# @tf.function
def train_one_step(model, opt, x, y, step, EPOCH):

    with tf.GradientTape() as tape:
        logits  = model(x)
        loss    = compute_loss(y, logits)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape,device_sparse='/cpu:0',device_dense='/cpu:0',compression=compression)
    grads = tape.gradient(loss, model.trainable_variables)


    opt.apply_gradients(zip(grads, model.trainable_variables))

    if step+EPOCH == 0:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    pred = tf.argmax(logits,axis=-1)
    compute_miou(y,pred)
    compute_accuracy(y, pred)

    return loss



# Sets up a timestamped log directory.
logdir = args.log_dir + "/" + str(IMG_SIZE) +'-' + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

def train(model, optimizer):
    train_ds = train_dataset
    iterations = tf.cast(len(image_list) // BATCH_SIZE, tf.int32)
    loss = 0.0
    accuracy = 0.0
    try:
        for EPOCH in range(0,EPOCHS):
            step = 0
            for x, y in train_ds:

                # if step == 10:
                #     tf.summary.trace_on(graph=True, profiler=True)

                loss = train_one_step(model, optimizer, x, y, step, EPOCH)

                if step % 10 == 0 and step > 0:

                    if hvd.rank() == 0:

                        # Training Prints
                        tf.print('Step', step, '/', iterations, ' of Epoch', EPOCH, ' Rank',
                                    hvd.local_rank(), ': loss', loss,'; accuracy', compute_accuracy.result())

                        compute_accuracy.reset_states()
                        # compute_miou.reset_states()
                        # filter images for images with tumor and non tumor


                        def filter_fn(image, mask):
                            return tf.math.zero_fraction(mask) > 0.2

                        for image, label in val_dataset.shuffle(buffer_size=100).take(1):
                            val_loss        = []
                            val_accuracy    = []
                            miou            = []
                            auc             = []
                            val_pred_logits   = model(image)
                            val_pred          = tf.math.argmax(val_pred_logits, axis=-1)
                            val_loss.append(compute_loss(label, val_pred_logits))
                            val_accuracy.append(compute_accuracy(label,val_pred))
                            # pdb.set_trace()
                            miou.append(compute_miou(label,val_pred))
                            val_pred = tf.expand_dims(val_pred,axis=-1)
                            auc.append(compute_auc(label, val_pred))


                            with file_writer.as_default():
                                # if step == 10:
                                #     tf.summary.trace_export(name="trace_%s_GPU"%IMG_SIZE,step=tf.cast(EPOCH*iterations+step,tf.int64),profiler_outdir=logdir)

                                image       = tf.cast(255 * image , tf.uint8)
                                mask        = tf.cast(255 * label , tf.uint8)
                                summary_predictions = tf.cast( val_pred * 255,tf.uint8)

                                # tf.summary.image('Image', image,                                            step=tf.cast(EPOCH*iterations+step,tf.int64),max_outputs=5)
                                # tf.summary.image('Mask',  mask ,                                            step=tf.cast(EPOCH*iterations+step,tf.int64),max_outputs=5)
                                # tf.summary.image('Prediction', summary_predictions,                         step=tf.cast(EPOCH*iterations+step,tf.int64),max_outputs=5)
                                tf.summary.scalar('Training Loss',loss,                                     step=tf.cast(EPOCH*iterations+step,tf.int64))
                                tf.summary.scalar('Training Accuracy',compute_accuracy.result(),            step=tf.cast(EPOCH*iterations+step,tf.int64))
                                tf.summary.scalar('Validation Loss',sum(val_loss)/len(val_loss),            step=tf.cast(EPOCH*iterations+step,tf.int64))
                                tf.summary.scalar('Validation Accuracy',sum(val_accuracy)/len(val_accuracy),step=tf.cast(EPOCH*iterations+step,tf.int64))
                                tf.summary.scalar('Mean IoU',sum(miou)/len(miou),                           step=tf.cast(EPOCH*iterations+step,tf.int64))
                                tf.summary.scalar('AUC',sum(auc)/len(auc),                                  step=tf.cast(EPOCH*iterations+step,tf.int64))

                                # Extract weights and filter out None elemens for aspp without weights
                                # weights = filter(None,[x.weights for x in model.layers])
                                # for var in weights:
                                #     tf.summary.histogram('%s'%var[0].name,var[0],step=tf.cast(EPOCH*iterations+step,tf.int64))

                            # Validation prints
                            tf.print('       ', 'val_loss', sum(val_loss) / len(val_loss),
                                        '; val_accuracy', sum(val_accuracy) / len(val_accuracy),
                                        '; mean_iou', sum(miou) / len(miou))
                    file_writer.flush()
                step += 1
                if EPOCH*iterations+step >= STEPS:
                    raise Exception
            if hvd.rank() == 0:
                model.save_weights(args.weight_path +"/Deeplabv3_level2_epoch"+str(EPOCH)+".h5")
    except:
        print('completed')
        pass
    if hvd.rank() == 0:
        model.save_weights(args.weight_path +'/Deeplabv3_level2_step%s.h5'%STEPS)


    return step, loss.numpy(), compute_miou.result().numpy()


# tf.summary.trace_on(graph=True, profiler=True)
step, loss, miou = train(model, opt)

print('Final step', step, ': loss', loss, '; miou', miou)

