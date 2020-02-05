from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2, 2), padding='same', 
                 input_shape=(10, 10, 1), strides=2)) # stride가 이미지 크기와 맞아떨어지지 않을 경우 맞는 부분만 계산(ex> (5, 5) => (2, 2))
model.add(Conv2D(100, (2, 2))) # Kernel의 크기는 input의 크기보다 작아야 한다!(같으면 안됨)
model.add(MaxPooling2D(2, 2))# MaxPooling은 최댓값이 겹치지 않도록(중요) stride를 알아서 조절해 데이터를 축소한다!(통상적으로 절반 ex> (6, 6) => (3, 3))
model.add(Flatten())
model.add(Dense(1))

model.summary()