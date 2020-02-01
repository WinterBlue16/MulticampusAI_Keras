
# dataset을 한번에 분리하는 함수(복습 필요)

from numpy import array
                        # dataset, 4
def split_sequence(sequence, n_steps):
    X,y = list(), list()
    for i in range(len(sequence)): # 10 # i는 0부터 시작된다!!
        end_ix = i + n_steps       # 0 + 4 = 4
        if end_ix > len(sequence)-1: # 4 > 9
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] # [0:4], [4] => 0, 1, 2, 3 / 4
        X.append(seq_x) # X 리스트에 추가 # 0, 1, 2, 3
        y.append(seq_y) # y 리스트에 추가 # 4
    return array(X), array(y) # numpy 배열로 만들기


dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for n_steps in range(1, 10):
    x, y = split_sequence(dataset, n_steps)
    print('n_steps :', n_steps)
    print(x)
    print(y)
    print()


