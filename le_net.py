from pyImageSearch.nn.conv.lenet import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from  keras import backend as k
from keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)

print ("[INFO] compiling model...")

optimizer = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
    metrics=["accuracy"])

print("[INFO] training network...")

H = model.fit(x_train, trainY, validation_data=(x_test, testY), batch_size=128,
    epochs=20, verbose=1)

print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=128, verbose=1)

print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]
))
