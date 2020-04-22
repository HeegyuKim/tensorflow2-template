# Tensorflow Template
케라스랑 텐서플로우로 매번 

## 핵심 기능
1. 파이썬으로는 데이터 로드랑 모델만 만들기
2. 각종 config 값은 yaml에 저장후 불러와서 사용
3. early stopping, tensorboard 같은 기능들 yaml에 간단하게 추가하기.
4. 반복 학습(iteration)을 이용해 여러 하이퍼파라미터로 모델 학습

# Requirements
```
python 3.6
tensorflow >= 2.0
```
# Setup

```bash
make

혹은

pip install -r requirements.txt
```

# 사용법

#### 1. YAML 파일에 학습에 필요한 파라미터 넣기 
```yaml
# yaml/mnist.yaml
module: mnist           # model/mnist.py
model: MnistClassifier  # mnist.py 의 MnistClassifier 클래스

train:  # 학습에 필요한 파라미터
    epochs: 20
    verbose: 1
    batch_size: 512
    validation_split: 0.1
    
    compile:    # model.compile()의 인자들과 동일
        optimizer: adam
        loss: sparse_categorical_crossentropy
        metrics: [accuracy]
    
    early_stopping: # tensorflow.keras.callbacks.EarlyStopping() 의 인자들과 동일
        monitor: loss
        mode: min
    
    tensorboard: true   # default false, true 면 logs/ 에 텐서보드 로깅
```

#### 2. 모델 클래스 구현
```python
# model/mnist.py 의 MnistClassifier 클래스
from . import BaseModel
from keras.datasets import mnist


class MnistClassifier(BaseModel):

    def load_data(self):
        """
            학습 데이터를 X, y 로 반환하면 됩니다.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train

    def build_model(self):
        """
            케라스 모델을 만들어서 리턴합니다.
        """
        model = Sequential([
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        return model
```

#### 3. 학습 실행
```bash
# yaml 파일이름을 넣음
python train.py mnist
```

## self.conf와 iteration 기능 사용하기
모델을 만들고 하이퍼파라미터를 바꿔가며 실행해보고싶을때가 있습니다.<br>
이럴때 iteration을 사용하면 됩니다
```yaml
# mnist-cnn.yaml
module: mnist
model: MnistCNNClassifier


train:
    epochs: 50
    verbose: 1
    batch_size: 512
    validation_split: 0.1
    
    # 학습에서 사용할 값들을 정의합니다.
    iteration: 3 # 총 세번 학습할 예정(기본 1)
    filters: [2, 4, 8]
    kernels: [3, 3, 3]

    compile:
        optimizer: adam
        loss: sparse_categorical_crossentropy
        metrics: [accuracy]

    early_stopping:
        monitor: loss
        mode: min
    
    summary: true       # default true
    tensorboard: true   # default false
        

test:
    checkpoint: mnist_best.hdf5
```

```python
# model/mnist.py

class MnistCNNClassifier(BaseModel):

    ...

    def build_model(self):
        """
            iteration 기능을 이용한 반복 학습.
            매 학습마다 다른 하이퍼파리미터를 지정할 수 있습니다.

            self.iter는 0부터 시작하고 학습이 반복될 때마다 1씩 증가합니다.
            self.conf 딕셔너리를 이용해서 yaml 파일 train: 의 값들을 가져올 수 있습니다.
        """
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
```


## 텐서보드
```bash
tensorboard --log_dir=logs
```
텐서모드 모델명은 `{module}-{model}-{iter}-{time}` 입니다. 

# TODO
1. 체크포인트 기능 지원(자동 저장/불러오기)
2. Validation 데이터 따로 넣을 수 있게.
3. model.evaluate(), model.predict() 지원
