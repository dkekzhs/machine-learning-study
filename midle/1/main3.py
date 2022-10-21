import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data_size = 100
perch_length = np.random.randint(80,440,(1,data_size))/10
perch_weight = perch_length**2-20*perch_length+160+np.random.randn(1,data_size)*80

train_input, test_input, train_target, test_target = train_test_split(
    perch_length.T, perch_weight.T, random_state=42)
print("학습데이터shape: ", train_input.shape, "테스트데이터shape: ", test_input.shape)

knr = KNeighborsRegressor() #학습 알고리즘
knr.fit(train_input, train_target) #컴퓨터가 학습 Mapping
print(knr.score(test_input, test_target))

test_prediction = knr.predict(test_input)
#테스트 세트에 대한 평균 절대값 오차 계산
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

print(knr.score(train_input, train_target))
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))