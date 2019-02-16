import matplotlib

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyImageSearch.nn.conv.mini_vgg_net import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.callbacks import LearningRateScheduler 


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="Path to the output loss/accuracy plot")
args = vars(ap.parse_args())

###learning rate will decrease by 0.25 after each 5 epochs:
def step_decay(epohs):

    initAlpha = 0.01
    factor=0.25
    dropEvery=5

    alpha = initAlpha*(factor**np.floor((1+epohs)/dropEvery))
    return float(alpha)



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

callbacks =[LearningRateScheduler(step_decay)]

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])



print("[INFO] training the network...")

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64,
    epochs=20, verbose=1,callbacks=callbacks)

print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=64)


print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=labelNames))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAr-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
