from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DataGenerator:
    def __init__(self,dbPath,batchSize,preprocessors=None,aug=None,binarizer=True,classes=2):

        #whetehr or not the labels are binarizer
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarizer=binarizer
        self.classes = classes

        #initialise the hdf5 database
        #determine the total number of entries 
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
    
    def generator(self,passes=np.inf):
        epochs=0

        while epoch<passes:
            for i in np.arrange(0,self.numImages,self.batchSize):
                images = self.db["images"][i:i+self.batchSize]
                labels= self.db["labels"][i:i+self.batchSize]

                if self.binarizer:
                    labels = np_utils.to_categorical(labels,self.classes)
                
                if self.preprocessors is not None:
                    #initialise the list of processed images :

                    imageProc = []
                    #loop over the images
                    for image in images:

                        #iterate over preprocessors:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        imageProc.append(image)
                    images = np.array(imageProc)
                ##not able to properly understand the meaninig for next and yield 
                if self.aug is not None:
                    (images,labels) = next(self.aug.flow(images,labels,batchSize=self.batchSize))
                yield(images,labels)
            
            epochs += 1
    
    def close(self):
        self.db.close()




