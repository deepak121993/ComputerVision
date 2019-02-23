from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array,load_img
from keras.applications import VGG16
from keras.applications import imagenet_utils
from pyImageSearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import progressbar


ap =  argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of faces")
ap.add_argument("-o", "--output", required=True,
    help="path to output hdf5 file")
ap.add_argument("-b", "--batch_size",type=int,default=32,
    help="size of batches")
ap.add_argument("-s", "--buffer_size",type=int,default=1000,
    help="size of buffer size")
args = vars(ap.parse_args())

bs = args["batch-size"]

print("[INFO] loading Images")
imagePaths =  list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

##extract the class label from the image path 

labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()
labels = le.fit_transform(labels)
print("labesls  " , labels)
print("[INFO] loading network")

model = VGG16(weights="imagenet",include_top=False)

dataset = HDF5DatasetWriter((len(imagePaths),512*7*7),args["output"],dataKey="features",\
                                    bufSize=args["buffer_size"])

dataset.storeClassLabels(le.classes_)

##initializes the progress bar :
widgets = ["Extracting Features:" , progressbar.Percentage(),"  ",progressbar.Bar(),"  ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()

##loop over images in patches ::

for i in np.arrange(0,len(imagePaths),bs):

    #extract the images and labels in batches
    ##then pass through the actual images frm the network and store the features 

    batch_paths = imagePaths[i:i+bs]
    batch_labels = labels[i:i+bs]
    batchImages =[]

    for (j,imagePath) in enumerate(batch_paths):
        ##load the images 
        ##and resize them to 224*224

        image = load_img(imagePath,target_size=(224,224))
        image = img_to_array(image)


        image = np.expand_dims(image,axis=0)

        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)
    

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages,batch_size=bs)

    features= features.reshape((features.shape[0],512*7*7))

    dataset.add(features,batch_labels)
    pbar.update(i)


dataset.close()
pbar.close()