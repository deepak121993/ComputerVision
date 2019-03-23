from deepergooglenet.config import tiny_imagenet_config as config
from pyImageSearch.utils.ranked import rank5_accuracy
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.dataset.simpleDatasetLoader import SimpleDatasetLoader
from pyImageSearch.io.hdf5datasetgenerator import HDF5DataGenerator
from keras.models import load_model
import json

means = json.load(open(config.DATASET_MEANS).read())

sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
iap = ImageToArrayProcessor()

testGen = HDF5DataGenerator(config.TEST_HDF5,64,aug=aug,preprocessors=[sp,mp,iap],\
                                classes=config.NUM_CLASSES)

print("[INFO] loading model")
model=load_model(config.MODEL_PATH)