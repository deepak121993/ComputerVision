import matplotlib 
matplotlib.use("Agg")

import matplotlib.pyplot as plt
#import the necessary packages 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyImageSearch.nn.conv.mini_vgg_net import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD
from keras.datasets import cifar10
import argparse
import os

ap =  argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to output directory")
ap.add_argument("-m", "--models", required=True,
    help="path to output models directory")
ap.add_argument("-n", "--num_models",type=int,default=5,
    help="size of batches")
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

aug = ImageDataGenerator(rotation_image=30,width_shift_range=0.1,height_shift_range=0.1,\
            horizontal_flip=True,fill_mode="nearest")

#loop over num of models to train::

for i in np.arange(0,args["num_models"]):

    print("[INFO] training model {}/{}".format(i+1,args["num-models"]))
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)

    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

    h = model.fit_generator(aug.flow(trainX,trainY,batch_size=64),epochs=10,\
    validation_data=(testX,testY),steps_per_epoch=len(trainX)//64,verbose=1)

    p=[args["models"],"model_{}.model".format(i)] 
    model.save(os.path.sep.join(p))

    predictions = model.predict(testX, batch_size=64)


    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=labelNames)
    
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=labelNames))

    p=[args["output"],"model_{}".format(i)]

    f=open(os.path.sep.join(p),"w")
    f.write(report)
    f.close()


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

