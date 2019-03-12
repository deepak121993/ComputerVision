from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.aspectawarepreprocess import AspectAwarePreprocessor
from pyImageSearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyImageSearch.preprocessing.patchpreprocessor import PatchPreporcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.preprocessing.croppreprocess import CropPreprocessor
from pyImageSearch.io.hdf5datasetgenerator import HDF5DataGenerator
from pyImageSearch.nn.conv.alexnet import AlexNet
from dog_vs_cat.config import dog_vs_cat_config as config
from pyImageSearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

means = json.loads(open(config.DATASET_MEAN).read())

#initailize the preprocessors:
sp =SimplePreprocessor(227,227)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
cp= CropPreprocessor(227,227)
iap = ImageToArrayProcessor()


#load the pretrained model
print("[INFO] loading thr pretrained model")
model = load_model( config.MODEL_PATH)

#predicting on test datawitout cropping :
testGen = HDF5DataGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],classes=2)
predictions = model.predict_generator(testGen.generator(),steps=testGen.numImages//64)

(rank1,_) = rank5_accuracy(predictions,testGen.db["labels"])
print("rank1 accuracy", rank1)

testGen.close()
##for crop _accuracy
testGen = HDF5DataGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],classes=2)

widget = ["building dataset",progressbar.Percentage()," ",progressbar.Bar()]
pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widget).start()

for (i,(images,labels)) in enumerate(testGen.generator(passes=1)):
    for image in images:
        crops = cp.preprocess(image)
        #seprate the crops and then convert them into array
        crops = np.array([iap.preprocess(c) for c in crops],dtype='float32')

        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))
    pbar.update(i)
pbar.finish()
(rank1,_) = rank5_accuracy(predictions,testGen.db["labels"])
print("rank1 accuracy", rank1)
testGen.close()

