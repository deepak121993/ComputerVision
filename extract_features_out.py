from keras.applications import ResNet50,imagenet_utils
from keras.preprocessing.image import img_to_array,load_img
from sklearn.preprocessing import LabelEncoder
from pyImageSearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import argparse
import progressbar
import random
import os


ap =  argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True,
    help="path to output models directory")
ap.add_argument("-o", "--output", required=True,
    help="path to output models directory")
ap.add_argument("-b", "--batch_size", default=16,
    help="path to output models directory")
ap.add_argument("-s", "--buffer_size",default=1000,
    help="path to output models directory")
args = vars(ap.parse_args())

bs = args["batch_size"]

print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep).split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading Network")
model = ResNet50(weights="imagenet",include_top=False)

#final pooling layer of resnet is 2048
dataset = HDF5DatasetWriter((len(imagePaths),2048),args["output"],dataKey="features",\
            bufSize=args["buffer_size"])

widget = ["building dataset",progressbar.Percentage()," ",progressbar.Bar()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widget).start()
print("length of images ",len(imagePaths)," batch size ",bs)
for i in np.arange(0,len(imagePaths),bs):

    batchPaths = imagePaths[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImages =[]

    for (j,imagePath) in enumerate(batchPaths):
        image  = load_img(imagePath,target_size=(224,224))
        image = img_to_array(image)

        #expand dims 
        image = np.expand(image,axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)
    
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages,batch_size=bs)
    features = features.reshape((features.shape[0],2048))
    print("here")
    dataset.add(features,batchLabels)
    pbar.update(i)
dataset.close()
pbar.finish()










