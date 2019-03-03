from dog_vs_cat.config import dog_vs_cat_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyImageSearch.preprocessing.aspectawarepreprocess import AspectAwarePreprocessor
from pyImageSearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np 
import json
import progressbar
import cv2
import os

trainPaths = list(paths.list_images(config.IMAGES_PATH))
print("paths " ,trainPaths , " length " ,len(trainPaths))
trainLabels = [p.split(os.sep.path)[2].split(".")[0] for p in trainPaths]
le = LabelEncoder()
trainLabels =le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
    random_state=42,stratify=trainLabels)
trainPaths,testPaths,trainLabels,testlabels=split

print("length of trainpaths 1",len(trainPaths))
##validation
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES,
    random_state=42,stratify=trainLabels)
trainPaths,valPaths,trainLabels,vallabels=split

print("length of trainpaths 2",len(trainPaths))

dataset = [("train",trainPaths,trainLabels,config.TRAIN_HDF5),
("test",testPaths,testLabels,config.TEST_HDF5),
("val",valPaths,vallabels,config.VAL_HDF5)
]

aap = AspectAwarePreprocessor(256,256)
(R,G,B)=([],[],[])


for (dtype,paths,labels,outputPath) in dataset:
    print("[INFO] building ..{}".format(outputPath))
    writer = HDF5DatasetWriter((len(paths),256,256,3),outputPath)

    widget = ["building dataset",progressbar.Percentage()," ",progressbar.Bar()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widget).start()


    for (i,(path,label)) in enumerate(zip(paths,labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dtype=="train":
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image],[label])
        pbar.update(i)
    

    pbar.finish()
    writer.close()

print("[INFO] serializing means ")

D={'R':np.mean(R),'G':np.mean(G),'B':np.mean(B)}
f= open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close()

