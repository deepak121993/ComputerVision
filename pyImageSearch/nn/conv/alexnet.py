from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.regularizers import l2


class AlexNet:

    @staticmethod
    def build(width,height,depth,classes,reg=0.002):
        #initalize the model with the input shape:

        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1

        #if we are using channel first then :

        if K.image_data_format() == "channel_first":
            inputShape = (depth,height,width)
            chanDim=1
        
        #first block
        model.add(Conv2D(96,kernel_size=(11, 11),strides=(4,4),
        kernel_regularizer=l2(reg),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
        model.add(Dropout(0.25))

        #second block
        model.add(Conv2D(256,kernel_size=(5,5),,
        kernel_regularizer=l2(reg),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
        model.add(Dropout(0.25))

        #third block

        model.add(Conv2D(384,kernel_size=(3,3),
        kernel_regularizer=l2(reg),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(96,kernel_size=(3, 3),
        kernel_regularizer=l2(reg),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(96,kernel_size=(3, 3),
        kernel_regularizer=l2(reg),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
        model.add(Dropout(0.25))

        #fourth block 
        #flatten
        model.add(Flatten())
        model.add(Dense(4096,kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        #fifth block

        model.add(Flatten())
        model.add(Dense(4096,kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        #final
        model.add(Dense(classes,kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model


