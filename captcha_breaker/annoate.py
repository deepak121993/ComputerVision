
from imutils import paths
import argparse
import imutils 
import cv2
import os


##to annoate the images :::


ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True)
ap.add_argument("-a","--annot",required=True)
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
counts={}


for (i,imagePath) in enumerate(imagePaths):
    print("[INFO] processing Image{}/{}".format(i+1,len(imagePaths)))
    
    try:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ##padding done here , it will pad the images with minimum 8 pixels in all directions

        gray = cv2.copyMakeBorder(gray,8,8,8,8,cv2.BORDER_REPLICATE)

        ##threshold the image to retrive the digits::

        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

        ##countours finds only 4 largest ones ::

        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:4]

        print("countours aka outlines ::",cnts)

        for c in cnts:

            ##compute the bounding box and then extract the digits::
            (x,y,w,h) = cv2.boundingRect(c)
            roi = gray[y-5:y+h+5,x-5:x+w+5]
            cv2.imshow("ROI",imutils.resize(roi,width=28))
            key = cv2.waitKey(0)


            if key==ord("*"):
                print("[INFO] ignoring this value")
                continue

            key = chr(key).upper()
            dirPath= os.path.sep.join([args["annot"],key])
            if not os.path.exists(dirPath):
                os.makedir(dirPath)

            count =count.get(key,1)
            p = os.path.sep.join([dirPath,"{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p,roi)

            ##increment the count of current key in captcha
            count[key] = count+1
    except KeyboardInterrupt:
        print("[ERROR] manually leaving script")
        break
    except Excetion as e:
        print("[INFO] skipping image due to ",e)




