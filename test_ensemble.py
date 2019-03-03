import matplotlib.pyplot as plt
#import the necessary packages 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD
from keras.datasets import cifar10
import argparse
import os
import numpy as np
import glob
import load_model


ap =  argparse.ArgumentParser()

ap.add_argument("-m", "--models", required=True,
    help="path to output models directory")
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

modelPaths = os.path.sep.join([args["models"],"*.models"])
modelPaths = list(glob.glob(modelPaths))
models=[]

for (i,modelPath) in enumerate(modelPaths):
    print("[INFO] loading models {}/{}".format(i+1,len(modelPaths)))

    model.append(load_model(x))

predictions=[]

for model in models:
    predictions.append(model.predict(testX,batch_size=64))
predictions= np.average(predictions,axis=0)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=labelNames))