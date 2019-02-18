from keras.preprocessing.image import img_to_array
from keras.models import load_model
from pyImageSearch.utils.captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input directory of images")
ap.add_argument("-m", "--model", required=True,
    help="path to input model")
args = vars(ap.parse_args())


model = load_model(args["model"])

##randomly sample the input images 
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths,size=(10,),replace=False)


for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray,20,20,20,20,cv2.BORDER_REPLICATE)

    #thresh = cv2.