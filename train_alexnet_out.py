from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.aspectawarepreprocess import AspectAwarePreprocessor
from pyImageSearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyImageSearch.preprocessing.patchpreprocessor import PatchPreporcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.callbacks.trainingmonitor import TrainingMonitor
from pyImageSearch.io.hdf5datagenerator import HDF5DataGenerator
from pyImageSearch.nn.conv.alexnet import AlexNet
from dog_vs_cat.config import dog_vs_cat_config as config
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,
                        shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

##initalize the image preprocessors
sp = SimplePreprocessor(227,227)
pp = PatchPreporcessor(227,227)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
iap = ImageToArrayProcessor()

trainGen = HDF5DataGenerator(config.TRAIN_HDF5,128,aug=aug,preprocessors=[sp,pp,mp,iap],classes=2)
#testGen  = HDF5DataGenerator(config.TEST_HDF5,128,aug=aug,preprocessors=[sp,pp,mp,iap],classes=2)
valGen   = HDF5DataGenerator(config.VAL_HDF5,128,aug=aug,preprocessors=[sp,pp,mp,iap],classes=2)
opt = Adam(lr=1e-3)   ##1e-3 == 0.001

model = AlexNet.build(width=227,height=227,depth=3,classes=2,reg=0.002)
model.compile(loss ="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

#contruct the path of callbacks 

path = os.path.sep.join([config.OUTPUT_PATH],"{}.png".format(os.getpid()))
callbacks = [TrainingMonitor(path)]

model.fit_generator(trainGen.generator(),steps_per_epoch= trainGen.numImages//128 ,
                    validation_data = valGen.generator(),validation_steps=valGen.numImages//128,
                    verbose=1,epochs=10,callbacks=callbacks)

model.save(config.MODEL_PATH,overwrite=True)

trainGen.close()
valGen.close()

