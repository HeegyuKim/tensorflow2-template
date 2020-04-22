"""
> python train.py vae.yaml

"""

import argparse
from utils import ssl_issue

from model import get_config, train

import tensorflow as tf
import keras


session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

parser = argparse.ArgumentParser()
parser.add_argument("yaml", help="get your training data")
args = parser.parse_args()

conf = get_config(args.yaml)
train(conf)
