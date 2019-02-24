from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.aspectawarepreprocess import AspectAwarePreprocessor
from pyImageSearch.dataset.simpleDatasetLoader import SimpleDatasetLoader
from pyImageSearch.nn.conv.fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
ap.add_argument("-m","--model",required=True)
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,\
height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print("len of images " , len(imagePaths))
print("class names ",classNames)
##initialize the preprocessing steps ::

aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayProcessor()

sdl = SimpleDatasetLoader(preprocessor=[iap])
(data,label) = sdl.load(imagePaths,verbose=500)
data = data.astype("float32")/255.0



# partition our data into training and test sets
(trainX, testX, trainY, testY) = train_test_split(data, label, test_size=0.25,
    random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")

baseModel = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel = FCHeadNet.build(baseModel,len(classNames),256)

model = Model(inputs=baseModel.input,outputs=headModel)


for layer in baseModel.layers:
    layer.trainable=False

print("[INFO] compiling model")
opt = RMSprop(lr=0.001)

model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training head ...")

model.fit_generator(aug.flow(trainX,trainY,batch_size=32),validation_data=(testX,testY) ,epochs=25,\
steps_per_epoch=len(trainX)//32,verbose=1)

print("[INFO] evaluating after initilization..")
predictions = model.predict(testX,batch_size=32)

print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

for layer in baseModel.layers[15:]:
    layer.trainable=True

print("[INFo] re-compiling model")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] fine tunining model ...")
model.fit_generator(aug.flow(trainX,trainY,batch_size=32),validation_data=(testX,testY),epochs=10,\
steps_per_epoch=len(trainX)//32,verbose=1)

print("[INFO] evaluating after fine-tuning..")
predictions = model.predict(testX,batch_size=32)

print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

print("[INFO] Serilizing model...")

model.save(args["model"])





