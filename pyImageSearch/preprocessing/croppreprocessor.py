import numpy as np 
import cv2

class CropPreprocessor:
    def __init__(self,width,height,horiz=True,inter=cv2.INTER_AREA):
        self.width = width
        self.height=height
        self.horiz=horiz
        self.inter=inter

    def preprocess(self,image):
        #initialize the list of crops
        crops=[]
        #grab the height and width of an image 
        #then use dimensions to define the corner of the image based

        (h,w) = image.shape[:2]
        coords = [[0,0,self.width,self.height],[w-self.width,0,w,self.height],
        [w-self.width,h-self.height,w,h],[0,h-self.height,self.width,h]]

        dW = int(0.5*(w-self.width))
        dH = int(0.5*(h-self.height))

        coords.append([dW,dH,w-dW,h-dH])

        for (startX,startY,endX,endY) in coords:
            crop = image[startY:endY,startX:endX]
            crop = cv2.resize(crop,(self.height,self.width),interpolation=self.inter)
            crops.append(crop)
        if self.horiz:
            #compute the horizontal flips for each crop 
            mirrors = [cv2.flip(c,1) for c in crops]
            crops.append(mirrors)
            
        #return the set of crops :::
        return np.array(crops)



