
IMAGES_PATH = "dataset/kaggle_dog_vs_cat/train"
NUM_CLASSES= 2
NUM_VAL_IMAGES= 300*NUM_CLASSES
NUM_TEST_IMAGES = 300*NUM_CLASSES

TRAIN_HDF5= "../dataset/kaggle_dog_vs_cat/hdf5/train.hdf5"
VAL_HDF5 = "../dataset/kaggle_dog_vs_cat/hdf5/val.hdf5"
TEST_HDF5 = "../dataset/kaggle_dog_vs_cat/hdf5/test.hdf5"

MODEL_PATH = "..output/alexnet_dog_vs_cat.model"
DATASET_MEAN = "../output/dog_vs_cat.json"
OUTPUT_PATH= "../output"