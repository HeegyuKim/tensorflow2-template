"""
python train.py vae.yaml

"""

import argparse
import os
import yaml
from utils import ssl_issue


parser = argparse.ArgumentParser()
parser.add_argument("yaml", help="get your training data")
args = parser.parse_args()


yaml_path = os.path.join("yaml", args.yaml + ".yaml")
with open(yaml_path) as f:
    conf = yaml.load(f)

mod = __import__("model.%s" % (conf['module'], ), fromlist=["model.%s" % (conf['module'], )])
model_class = getattr(mod, conf['model'])

model = model_class()
model.setup(conf)
model.train(conf['train'])