import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from . import BaseModel
from keras.datasets import mnist


class MnistClassifier(BaseModel):

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train

    def build_model(self):
        model = Sequential([
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        return model
        

class MnistCNNClassifier(BaseModel):

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows=x_train[0].shape[0]
        img_cols=x_test[0].shape[1]

        x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
        x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test

        print(x_train.shape)

        return x_train, y_train

    def build_model(self):
        i = self.iter
        filters = self.conf["filters"][i]
        kernels = self.conf["kernels"][i]

        model = Sequential([
            Conv2D(filters, (kernels, kernels), input_shape=(28, 28, 1)),
            MaxPool2D(2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        return model
        