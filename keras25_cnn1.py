from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2, 2), padding='same', 
                 input_shape=(5, 5, 1)))
model.add(MaxPooling2D(3, 3))
model.add(Flatten())
model.add(Dense(1))

model.summary()