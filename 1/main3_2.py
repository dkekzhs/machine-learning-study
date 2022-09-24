import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data_size = 100
perch_length = np.random.randint(80,440,(1,data_size))/10
perch_weight = perch_length**2-20*perch_length+160+np.random.randn(1,data_size)*80

train_input, test_input, train_target, test_target = train_test_split(
    perch_length.T, perch_weight.T, random_state=42)

knr = KNeighborsRegressor(n_neighbors=3) #학습 알고리즘
knr.fit(train_input, train_target) #컴퓨터가 학습 Mapping

print(knr.predict([[50]])) #예측 못 함
distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(50, 1033, marker='^')
plt.show()

#선형 회귀
lr = LinearRegression()
lr.fit(train_input, train_target)
print(knr.predict([[50]]))
print(lr.predict([[50]]))
