from deepergooglenet.config import tiny_imagenet_config as config
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.aspectawarepreprocess import AspectAwarePreprocessor
from pyImageSearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyImageSearch.preprocessing.patchpreprocessor import PatchPreporcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.callbacks.trainingmonitor import TrainingMonitor
from pyImageSearch.io.hdf5datasetgenerator import HDF5DataGenerator
from pyImageSearch.nn.conv.deepergooglenet import DeeperGoogLeNet
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as k
import argparse
import json

ap =  argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
    help="path to output directory")
ap.add_argument("-m", "--model", type=str,
    help="path to output models directory")
ap.add_argument("-s", "--start-epoch",type=int,default=0,
    help="size of batches")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,\
            shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

means = json.load(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
iap = ImageToArrayProcessor()

trainGen = HDF5DataGenerator(config.TRAIN_HDF5,64,aug=aug,preprocessors=[sp,mp,iap],\
                                classes=config.NUM_CLASSES)
valGen = HDF5DataGenerator(config.VAL_HDF5,64,aug=aug,preprocessors=[sp,mp,iap],\
                                classes=config.NUM_CLASSES)

if(args["model"] is None):
    print("[INFO] compiling model ")
    model= DeeperGoogLeNet.build(width=64,height=64,depth=3,classes=config.NUM_CLASSES,reg=0.0002)
    opt=Adam(1e-3)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

else:
    print("[INFO] loading model {}".format(args["model"]))
    model = load_model(args["model"])
    print("old lr {}".format(k.get_value(model.optimizer.lr)))
    k.set_value(model.optimizer.lr,1e-5)
    print("New Lr {}".format(k.get_value(model.optimizer.lr)))

fname=os.path.sep.join([args["weight"],"weight-{epoch:03d}-{val_loss:.4f}.hdf5"])
callbacks=[TrainingMonitor(config.FIG_PATH,jsonPath=config.JSON_PATH,startAt=args["start-epoch"]),\
ModelCheckpoint(fname,monitor="val_loss",mode="min",save_best_only=True)]

model.fit_generator(trainGen.generator(),steps_per_epoch=trainGen.numImages//64,\
validation_data=valGen.generator(),validation_steps=valGen.numImages // 64,epochs=10,max_queue_size=64*2\
callbacks=callbacks,verbose=1)

trainGen.close()
valGen.close()

