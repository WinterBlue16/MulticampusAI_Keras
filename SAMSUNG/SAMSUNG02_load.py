import numpy as np
import pandas as pd

# 3. 저장한 np.array 파일 불러오기
samsung = np.load('./SAMSUNG/DATA/Samsung.npy')
kospi200 = np.load('./SAMSUNG/DATA/kospi200.npy')

# print(samsung)
# print(samsung.shape)
print(kospi200)
print(kospi200.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps # x의 끝 값(몇 개씩 자를 것인가)
        y_end_number = x_end_number + y_column # y의 시작값
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] # x의 위치 => 행 지정, 열 전부
        tmp_y = dataset[x_end_number:y_end_number, 3] # y의 위치, 숫자는 '종가'의 idx
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x), np.array(y)

x, y =split_xy5(samsung, 5, 1)
print(x.shape)
print(y.shape)
print(x[0,:], '\n', y[0])