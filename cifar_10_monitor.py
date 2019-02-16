import matplotlib
matplotlib.use("Agg")

from pyImageSearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyImageSearch.nn.conv.mini_vgg_net import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os



ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True)
args = vars(ap.parse_args())

print("[INFO] Process ID :{}".format(os.getpid()))



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
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

##construct the set of callbacks

figPath = os.path.sep.join([args["output"],".png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"],"{}.json".format(os.getpid())])
callbacks=[TrainingMonitor(figPath,jsonPath=jsonPath)]
print("[INFO] training the network...")


model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64,
    epochs=10, verbose=1,callbacks=callbacks)

