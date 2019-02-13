from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.dataset.simpleDatasetLoader import SimpleDatasetLoader
from pyImageSearch.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
ap.add_argument("-m","--model",required=True)
args = vars(ap.parse_args())


print("[INFO] going to load images")
image_path = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32,32)
iap = ImageToArrayProcessor()

sdl = SimpleDatasetLoader(preprocessor=[sp,iap])
(data,label) =sdl.load(image_path,verbose=500)

data = data.astype("float32")/255.0

# partition our data into training and test sets
(trainX, testX, trainY, testY) = train_test_split(data, label, test_size=0.25,
    random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")

# initialize stochastic gradient descent with learning rate of 0.005
opt = SGD(lr=0.005)

model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32,
    epochs=100, verbose=1)
print("path to save " ,args["model"])
model.save(args["model"])

print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=32)

print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["cat", "dog", "panda"]
))
