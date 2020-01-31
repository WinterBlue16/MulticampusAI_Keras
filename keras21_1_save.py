# 모델 만들어 저장하기

import numpy as np

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

# model.add(Dense(512, input_dim=1)) # 레이어 추가
model.add(Dense(512, input_shape=(3,))) # input_shape = (1,), 벡터가 1개라는 뜻
model.add(Dense(256)) # Node 조절 
model.add(Dense(256)) 
model.add(Dense(1))

model.save('./save/savetest01.h5')
print('저장 완료')