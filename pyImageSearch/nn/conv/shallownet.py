from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation,Flatten,Dense
from keras import backend as K 



class ShallowNet:
    @staticmethod
    def build(width,height,depth,classes):

        model = Sequential()
        input_shape = (width,height,depth)

        if K.image_data_format() =="channel_first":
            input_shape=(depth,width,height)
        
        model.add(Conv2D(32,(3,3),padding="same",input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))


        return model
