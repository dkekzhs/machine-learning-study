import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

data_size=100
perch_length=np.random.randint(80,440,(1,data_size))/10 #(1,100)
perch_weight=perch_length**2-20*perch_length+110+np.random.randn(1,data_size)*50 #(1,100)
# perch_weight=np.random.randint(80,440,(1,data_size))/10 #(1,100)
#결과 1

train_input, test_input, train_target, test_target = train_test_split(
perch_length.T, perch_weight.T, random_state=42)
test_input = test_input.reshape(-1,1)
train_input = train_input.reshape(-1,1)


# print("학습데이터Shape: ", train_input.shape,"테스트데이터Shape: ", test_input.shape)

# knr = KNeighborsRegressor()
# knr.fit(train_input,train_target)
# print(knr.score(test_input,test_target))

# test_prediction = knr.predict(test_input)
# mae = mean_absolute_error(test_target,test_prediction)
# print(mae)

lr = LinearRegression()
lr.fit(train_input, train_target)

print(lr.coef_, lr.intercept_)
plt.scatter(train_input, train_target)
plt.plot([0, 60], [float(0*lr.coef_+lr.intercept_), float(60*lr.coef_+lr.intercept_)])
plt.scatter(60, lr.predict([[60]]), marker='^')
plt.show()
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
