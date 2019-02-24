import h5py
import os

class HDF5DatasetWriter:

    def __init__(self,dims,outputPath,dataKey="images",bufSize=1000):

        #check to see if the path is exists and if not then raise an exception 

        if os.path.exists(outputPath):
            raise ValueError("supplied path is already exists  can't overidden",outputPath)

        #opn the HDF5 dataset and create 2 two datasets
        #one for image features nd other for labels


        self.db = h5py.File(outputPath,"w")
        self.data = self.db.create_dataset(dataKey,dims,dtype="float")
        self.labels = self.db.create_dataset("labels",(dims[0],),dtype='int')

        self.bufSize = bufSize
        self.buffer = {"data":[],"labels":[]}
        self.idx=0

    def add(self,rows,labels):

        self.buffer["data"].extends(rows)
        self.buffer["labels"].extends(labels)

        if len(self.buffer["data"]) > self.bufSize:
            self.flush()
    
    def flush(self):
        
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]

        self.idx = i
        self.buffer = {"data":[],"labels":[]}

    def storeClassLabels(self,classLabels):

        #create a dataset which contains the actual class label names
        dt = h5py.special_dtype(vlen="str")
        labelSet = self.db.create_dataset("label_names",len((classLabels)),dtype=dt)

        labelSet[:] = classLabels

    def close(self):

        if len(self.buffer["data"])> 0:
            self.flush()

        self.db.close()
