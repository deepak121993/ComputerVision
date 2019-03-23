from deepergooglenet.config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyImageSearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os



trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le =    ()

trainLabels = le.fit_transform(trainLabels)
split = train_test_split(trainPaths,trainLabels,test_size=config.NUM_TEST_IMAGES,stratify=trainLabels,\
        random_state=42)
(trainPaths,testPaths,trainLabels,testLabels)=split

M = open(config.VAL_MAPPING).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]

valPaths = [os.path.join([config.VAL_IMAGES,m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])

datasets = [("train",trainPaths,trainLabels,config.TRAIN_HDF5),("val",valPaths,valLabels,config.VAL_HDF5),\
("test",testPaths,testLabels,config.TEST_HDF5)]

(R,G,B)= ([],[],[])

for (dtype,paths,labels,outputPath) in datasets:
    print("[INFO] building {}..".format(outputPath))
    writer = HDF5DatasetWriter((len(paths),64,64,3),outputPath)

    widget = ["building dataset",progressbar.Percentage()," ",progressbar.Bar()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widget).start()

    for (i,(path,label)) in enumerate(zip(  paths,labels)):
        image = cv2.imread(path)

        if(dtype=="train"):
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image],[label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("[INFO] Serializing means...")
D={"R":np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f = open(config.DATASET_MEAN,"w") 
f.write(json.dumps(D))
f.close()

    




