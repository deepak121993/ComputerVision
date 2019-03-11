import cv2 

class MeanPreprocessor:

    def __init__(self,rMean,gMean,bMean):
        self.rMean = rMean
        self.gMean=gMean
        self.bMean=bMean

    def preprocess(self,image):
        #split the image into their respective r , g , b
        (B,G,R) = cv2.split(image.astype("float32"))

        R -= self.rMean
        B -= self.bMean
        G -= self.gMean
        #merge the channel back together and return the image
        return cv2.merge([B,G,R])