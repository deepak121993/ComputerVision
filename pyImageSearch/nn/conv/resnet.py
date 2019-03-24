from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D,ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input,add
from keras.models import Model
from keras.layers import concatenate 

from keras.regularizers import  l2

class ResNet:

    @staticmethod
    def residual_module(data,K,stride,chanDim,red=False,reg=0.0001,bnEps=2e-5,bnMom=0.9):

        shortcut=data

        #first block of resnet is 1x1 convs 
        bn1= BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(K*0.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1)

        #second block of resnet modules are of 3x3 conv
        bn2 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 =  Conv2D(int(K*0.25),(3,3),use_bias=False,padding="same",kernel_regularizer=l2(reg),strides=stride)(act2)

        #final bottleneck  another 1x1 filters
        bn3 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 =  Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)

        if red:
            
            shortcut = Conv2D(K,(1,1),strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)

        x = add([conv3,shortcut])
        return x
    
    @staticmethod 
    def build(width,height,depth,classes,stages,filters,reg=0.0001,bnEps=2e-5,bnMom=0.9,dataset="cifar"):
        inputShape = (height,width,depth)
        chanDim=-1

        input = Input(shape=inputShape)
        x= BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(input)

        #check if the dataset is cifar
        if(dataset=="cifar"):
            x = Conv2D(filters[0],(3,3),use_bias=False,padding="same",kernel_regularizer=l2(reg))(x)
        
        for i in range(0,len(stages)):

            stride = (1,1) if(i==0) else (2,2)
            x= ResNet.residual_module(x,filters[i+1],stride,chanDim,True,reg,bnEps,bnMom)

            for j in range(0,stages[i]-1):
                x = ResNet.residual_module(x,filters[i+1],(1,1),chanDim,True,reg,bnEps,bnMom)


        x= BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnEps)(x)
        x= Activation("relu")(x)
        x= AveragePooling2D((8,8))(x)

        #softmax classsifier
        
        x=  Flatten()(x)
        x= Dense(classes,kernel_regularizer=l2(reg))(x)
        x= Activation("softmax")(x)

        model = Model(input,x,name="resnet")

        return model

    
