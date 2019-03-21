from os import path

#define the paths of train and validation directories

TRAIN_IMAGES = "dataset/tiny-imagenet-200/train"
VAL_IMAGES = "dataset/tiny-imagenet-200/val/images"

VAL_MAPPING = "dataset/tiny-imagenet-200/val/val_annotations.txt"
WORDNET_IDS = "dataset/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "dataset/tiny-imagenet-200/words.txt"

NUM_CLASSES=200
NUM_TEST_IMAGES =50*NUM_CLASSES

#paths of datasets 

TRAIN_HDF5="dataset/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "dataset/tiny-imagenet-200/hdf5/test.hdf5"
TEST_HDF5 ="dataset/tiny-imagenet-200/hdf5/val.hdf5"

DATASET_MEAN = "deepergooglenet/output/tiny_imagenet_mean.json"
OUTPUT_PATH  = "deepergooglenet/output"
MODEL_PATH = path.sep.join([OUTPUT_PATH,"checkpoints/epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH,"deepergooglenet_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH,"deepergooglenet_tinyimagenet.json"])

