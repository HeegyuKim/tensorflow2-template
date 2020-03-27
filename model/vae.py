import os

from . import BaseModel
import data.celeba

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

# from utils.callbacks import CustomCallback, step_decay_schedule 



class VAE(BaseModel):


    def setup(self, conf):
        self.data = data.celeba.CelebAData(**conf['train'])


    def build(self, conf):
        encoder_input = Input(shape=self.data.input_dim, name='encoder_input')
        x = encoder_input
        enc = conf['encoder']
        z_dim = conf['z_dim']

        for i in range(enc['n_of_layers']):
            conv_layer = Conv2D(
                filters = enc['filters'][i]
                , kernel_size = enc['kernel_sizes'][i]
                , strides = enc['strides'][i]
                , padding = 'same'
                , name = 'encoder_conv_' + str(i)
                )

            x = conv_layer(x)

            if conf['use_batch_norm']:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if conf['use_dropout']:
                x = Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(z_dim, name='mu')(x)
        self.log_var = Dense(z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        
        ### THE DECODER
        dec = conf['decoder']
        decoder_input = Input(shape=(z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(dec['n_of_layers']):
            conv_t_layer = Conv2DTranspose(
                filters = dec['filters'][i]
                , kernel_size = dec['kernel_sizes'][i]
                , strides = dec['strides'][i]
                , padding = 'same'
                , name = 'decoder_conv_t_' + str(i)
                )

            x = conv_t_layer(x)

            if i < dec['n_of_layers'] - 1:
                if conf['use_batch_norm']:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if conf['use_dropout']:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

            

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)        

        print(self.encoder.summary())
        print(self.decoder.summary())

    def compile(self, conf):
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            return conf['r_loss_factor'] * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + kl_loss

        optimizer = Adam(lr=conf['learning_rate'])
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss], experimental_run_tf_function=False)

    def train(self, conf): 
        self.build(conf)
        self.compile(conf)

        run_folder = conf['checkpoint_dir']
        print_every_n_batches = conf.get('print_every_n_batches', 10)

        # custom_callback = CustomCallback(run_folder, print_every_n_batches, 0, self)
        # lr_sched = step_decay_schedule(initial_lr=conf['learning_rate'], decay_factor=1, step_size=1)

        checkpoint_filepath=os.path.join(run_folder, "weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights.h5'), save_weights_only = True, verbose=1)

        # callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]
        callbacks_list = [checkpoint1, checkpoint2]

        self.model.save_weights(os.path.join(run_folder, 'weights.h5'))
            
        epochs = conf['epochs']

        self.model.fit_generator(
            self.data.data_flow
            , shuffle = True
            , epochs = epochs
            , initial_epoch = 0
            , callbacks = callbacks_list
            , steps_per_epoch=self.data.steps_per_epoch 
            )


    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)

