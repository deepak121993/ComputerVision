from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report 
from pyImageSearch.preprocessing.simpleProcessor import SimplePreprocessor
from pyImageSearch.dataset.simpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-k","--neighbors",type=int,default=1)
ap.add_argument("-j","--jobs",type =int,default=1)
args = vars(ap.parse_args())


print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))


sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessor=[sp])

(data,label) = sdl.load(imagePaths,verbose=500)
data = data.reshape((data.shape[0],3072))


print("[INFO]  feature matrix : {:.1f}MB".format(data.nbytes/1024*1000.0))

le = LabelEncoder()
labels = le.fit_transform(label)

(x_train,x_test,y_train,y_test) = train_test_split(data,labels,test_size=0.25,random_state=40)


print("INFO loading model")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(x_train,y_train)


print(classification_report(y_test,model.predict(x_test)),target_names=le.classes_)
