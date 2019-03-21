import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyImageSearch.nn.conv.minigooglenet import MiniGoogleNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
import argparse
import os
import numpy as np


NUM_EPOCHS = 20
INIT_LR =  5e-3

def poly_decay(epochs):
    maxEpochs=NUM_EPOCHS
    base_LR = INIT_LR
    power = 1.0

    alpha = base_LR*(1-(epochs/float(maxEpochs)))**power

    return alpha

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True)
ap.add_argument("-m","--model",required=True)
args = vars(ap.parse_args())



print("[INFO] loading CIFAR-10 data...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "birds", "cat", "deer", "dog", "frog",
     "horse", "ship", "truck"]

print("[INFO] compiling model") 


