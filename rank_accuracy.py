from pyImageSearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py


ap = argparse.ArgumentParser()
ap.add_argument("-d","--db",required=True,help="path to hd5 file")
ap.add_argument("-m","--model",required=True)
ap.add_argument("-j","--jobs",type =int,default=-1)
args = vars(ap.parse_args())

print("[INFO] loading model")
model = pickle.loads(open(args["model"],"rb").read())


db = h5py.File(args["db"],"r")
i= int(db['labels'].shape[0]*0.75)

print("[INFO] predicting")

preds = model.predict_proba(db["features"][i:])
(rank1,rank5) = rank5_accuracy(preds,db["labels"][i:])

print("[INFO] rank1 ",rank1*100)
print("[INFo] rank5 ",rank5*100)

db.close()