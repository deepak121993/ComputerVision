from deepergooglenet.config import tiny_imagenet_config as config
from pyImageSearch.utils.ranked import rank5_accuracy
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.dataset.simpleDatasetLoader import SimpleDatasetLoader
from pyImageSearch.io.hdf5datasetgenerator import HDF5DataGenerator
from keras.models import load_model
import json

means = json.loads(open(config.DATASET_MEANS).read())

sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
iap = ImageToArrayProcessor()

testGen = HDF5DataGenerator(config.TEST_HDF5,64,aug=aug,preprocessors=[sp,mp,iap],\
                                classes=config.NUM_CLASSES)

print("[INFO] loading model")
model=load_model(config.MODEL_PATH)

print("[INFO] prediction on test Data ")
predictions = model.predict_generator(testGen.generator(),steps=testGen.numImages // 64,max_queue_size=64*2)

(rank1,rank5) = rank5_accuracy(predictions,testGen.db["labels"])
print("[INFO] rank-1 :{:.2f}".format(rank1*100))
print("[INFO] rank-5 :{:.2f}".format(rank5*100))

testGen.close()


