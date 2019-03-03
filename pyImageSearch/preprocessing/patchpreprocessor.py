from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreporcessor:
    def __init__(self,width,height):

        self.width = width
        self.height= height

    def preprocess(self,image):
        #extract a random crop from image with the target width and height
        return extract_patches_2d(image,(self.height,self.width),max_patches=1)[0]