import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from . import BaseModel
import data.mnist


class MnistClassifier(BaseModel):

    def setup(self, conf):
        (x_train, y_train), (x_test, y_test) = data.mnist.load()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def build(self, conf):
        model = Sequential([
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        return model

    def train(self, conf):
        model = self.build(conf)
        checkpoint = ModelCheckpoint(
            "checkpoints/mnist_best.hdf5",
            monitor='loss',
            verbose=1,
            save_best_only=True, 
            mode='auto', 
            period=1
        )
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        model.fit(
            self.x_train, 
            self.y_train,
            epochs=conf['epochs'],
            callbacks=[checkpoint, es],
            validation_split=0.2
            )
    
    def test(self, conf):
        pass

