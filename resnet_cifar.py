import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyImageSearch.nn.conv.resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
from pyImageSearch.callbacks.trainingmonitor import TrainingMonitor
from keras.optimizers import RMSprop,SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.models import load_model
from keras import backend as k
import argparse
import os
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-o","--checkpoints",required=True)
ap.add_argument("-m","--model",type=str)
ap.add_argument("-s","--start-epoch",type=int,default=0)
args = vars(ap.parse_args())


print("[INFO] loading CIFAR-10 data...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX,axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "birds", "cat", "deer", "dog", "frog",
     "horse", "ship", "truck"]


aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,\
            shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

if(args["model"] is None):
    print("[INFO] compiling model")
    opt =SGD(lr=1e-1)
    model= ResNet.build(32,32,3,10,(9,9,9),(64,64,128,256),reg=0.0005)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
else:
    print("[INFO] loading model")
    model = load_model(args["model"])
    print("old lr {}".format(k.get_value(model.optimizer.lr)))
    k.set_value(model.optimizer.lr,1e-5)
    print("New Lr {}".format(k.get_value(model.optimizer.lr)))

fname=os.path.sep.join([args["checkpoints"],"weight-{epoch:03d}-{val_loss:.4f}.hdf5"])
callbacks=[TrainingMonitor(config.FIG_PATH,jsonPath=config.JSON_PATH,startAt=args["start_epoch"]),\
ModelCheckpoint(fname,monitor="val_loss",mode="min",save_best_only=True)]

model.fit_generator(aug.flow(trainX,trainY,batch_size=128),validation_data=(testX,testY),\
steps_per_epoch=len(trainX)//128,epochs=10,callbacks=callbacks,verbose=1)