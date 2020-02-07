# RCV => GridSearch로 변경

# 0. 라이브러리 불러오기 
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np

# 1. 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train) # One hot encodiong
Y_test = np_utils.to_categorical(Y_test)

# 2. 함수 만들기
def bulid_network(keep_prob=0.5, optimizer='adam'): # dropout = 0.5, optimizer = 'adam'
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                  metrics=['acc'])
    return model

def create_hyperparameters(): # 값만 넣어놓음
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout= np.linspace(0.1, 0.5, 5)
    return[{"batch_size":batches, 
           "optimizer":optimizers, 
           "keep_prob":dropout}]
 
# 3. keras & sklearn wrapping(=싸다)   
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor # keras에서 sklearn을 사용한다! # KerasRegressor도 적용해보기
model = KerasClassifier(build_fn=bulid_network, verbose=1) # keras의 함수형 모델을 가져오겠다!

hyperparameters = create_hyperparameters() # 값만 넣어놓음
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = GridSearchCV(model, hyperparameters, cv=3, verbose=1) # cv = crossvalidation 수 # 안 멈추는 문제 발생
search.fit(X_train, Y_train)

print(search.best_params_) # {'optimizer': 'adadelta', 'keep_prob': 0.1, 'batch_size': 40}


