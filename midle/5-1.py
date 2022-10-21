import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

fish = pd.read_csv("https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv")

fish_input = fish[["Weight","Length","Diagonal","Height","Width"]].to_numpy()
fish_target = fish["Species"].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target
, test_size=0.2, random_state=42)


kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_input,train_target)

print(kn.classes_)
print(kn.predict(fish_input[:5]))

proba = kn.predict_proba(fish_input[:5])
print(np.round(proba,decimals=4))

poly =PolynomialFeatures(degree=1)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
test_ploy = poly.transform(test_input)


ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_ploy)

bream_smelt_indexes = (train_target == "Bream") | (train_target == "Smelt")
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

print(lr.coef_,lr.intercept_)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))