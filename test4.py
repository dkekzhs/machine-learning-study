import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
df = pd.read_csv("./HOME_전력거래_정산단가_연료원별.csv" ,encoding="cp949")
df2 = pd.read_csv("./HOME_전력거래_계통한계가격_가중평균SMP.csv",encoding="cp949")

print(df,df2)
z1 = df.to_numpy()
z1 = np.delete(z1,1,axis=1)
z1 = np.delete(z1,0,axis=1)
z2 = df2.to_numpy()
z2 = np.delete(z2,0,axis=1)

train_input, test_input, train_target, test_target = train_test_split(z1, z2
, test_size=0.2, random_state=42)

poly =PolynomialFeatures(include_bias=False, degree=3)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_ploy = poly.transform(test_input)
t1_poly = poly.transform(np.array([[60.89975127,189.7558566,214.2727383,611.5256006,274.2694919]]))

lr = LinearRegression()
lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_ploy,test_target))
print("2022/가격예측",lr.predict(t1_poly))
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_ploy)
t1_scaled = ss.transform(t1_poly)

alpha_list= [0.0001,0.001,0.01,0.1,1,10,100,1000]

for alpha in alpha_list:
    ridge = Ridge(alpha = alpha)
    ridge.fit(train_scaled,train_target)

    print("alpha = ",alpha ," train = ",ridge.score(train_scaled,train_target))
    print("test = ",ridge.score(test_scaled,test_target))
    print("2022가격예측 : ",ridge.predict(t1_scaled))