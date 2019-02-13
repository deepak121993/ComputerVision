from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyImageSearch.preprocessing.imagetoarrayprocessor import ImageToArrayProcessor
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.dataset.simpleDatasetLoader import SimpleDatasetLoader
from pyImageSearch.nn.conv.shallownet import ShallowNet
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
ap.add_argument("--m","--model",required=True)
args = vars(ap.parse_args())

classLabels = ["cat","dog","panda"]

print("going to load images")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idx = np.random.randint(0,len(imagePaths),size=(10,))
imagesPaths = imagePaths[idx]


sp = SimplePreprocessor(32,32)
iap = ImageToArrayProcessor()

sdl = SimpleDatasetLoader(preprocessor=[sp,iap])
(data,label) =sdl.load(image_path,verbose=500)

data = data.astype("float32")/255.0



print("loading models ...")
model = load_model(args["model"])

print("predicting ")
preds=model.predict(data,batch_size=32).argmax(axis=1)


for i , imagePath in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image,"label: {}".format(classLabels[preds[i]]),
    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7)
    cv2.imshow("Image",image)
    cv2.waitkey(0)
