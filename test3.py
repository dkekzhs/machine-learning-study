import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
df = pd.read_csv("./HOME_전력거래_정산단가_연료원별.csv")
df2 = pd.read_csv("./HOME_전력거래_계통한계가격_가중평균SMP.csv")

print(df,df2)
z1 = df.to_numpy()
z2 = df2.to_numpy()
z3 = np.array([60.89975127,189.7558566,214.2727383,611.5256006,274.2694919])

train_input, test_input, train_target, test_target = train_test_split(z1, z2
, test_size=0.2, random_state=42)

poly =PolynomialFeatures(include_bias=False, degree=2)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_ploy = poly.transform(test_input)


lr = LinearRegression()
lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_ploy,test_target))

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_ploy)
test2_scaled = ss.transform(z3)

ridge = Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))
print(ridge.predict(test2_scaled))


