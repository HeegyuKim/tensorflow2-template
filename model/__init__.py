

import os
import yaml

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from time import time



class BaseModel:

    def __init__(self, conf):
        self.name = f"{conf['module']}-{conf['model']}"
        self.conf = conf["train"]
        self.max_iteration = self.conf.get("iteration", 1)
        self.iter = 0
        
    def load_data(self):
        """
            X, y = model.load_data()

            # Validation이 존재한다면
            X, y, validation_x, validation_y = model.load_data()
        """
        pass
        
    def create_model(self):
        model = self.build_model()
        
        compile_args = self.get_compile_args()
        
        model.compile(**compile_args)

        if self.conf.get("summary", True):
            print(model.summary())

        return model

    def build_model(self):
        """
            Keras Model을 반환해야합니다.
        """
        pass
        
    def get_fitting_args(self):
        x, y = self.load_data()

        if x is None or y is None:
            raise Exception("학습 데이터가 준비되지 않았습니다.")

        args = {
            "x": x,
            "y": y,
            "verbose": self.get_conf('verbose', 0),
            'epochs': self.get_conf('epochs', required=True),
            'batch_size': self.get_conf('batch_size', None),
            'validation_split': self.get_conf('validation_split', None),
        }

        args['callbacks'] = self.generate_callbacks()

        return args

    def get_conf(self, key, default=None, message = None, required=False):
        if key in self.conf:
            return self.conf[key]
        elif required:
            raise Exception(message)
        else:
            return None
    
    def generate_callbacks(self):
        callbacks = []

        if "early_stopping" in self.conf:
            kwargs = self.conf["early_stopping"]
            callbacks.append(EarlyStopping(**kwargs))

        if self.conf.get("tensorboard", False):
            log_dir = f"logs/{self.name}"
            if self.max_iteration > 1:
                log_dir = log_dir + f"-{self.iter}"
            log_dir = log_dir + str(time())

            callbacks.append(TensorBoard(log_dir=log_dir))

        return callbacks

    def get_compile_args(self):
        args = self.conf['compile']
        if "optimizer" not in args:
            raise Exception("compile.optimizer 가 정의되지 않았습니다.")
        if "loss" not in args:
            raise Exception("compile.loss 가 정의되지 않았습니다.")
        return args


    def train(self):
        for i in range(self.max_iteration):
            self.iter = i
            kwargs = self.get_fitting_args()
            model = self.create_model()
            model.fit(**kwargs)


def get_config(yaml_file):
    yaml_path = os.path.join("yaml", yaml_file + ".yaml")
    with open(yaml_path) as f:
        conf = yaml.load(f)
    return conf


def train(conf):
    mod = __import__("model.%s" % (conf['module'], ), fromlist=["model.%s" % (conf['module'], )])
    model_class = getattr(mod, conf['model'])

    model = model_class(conf)
    model.train()