import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

fish = pd.read_csv('midterm.csv')
fish_input = fish[['Height', 'Diagonal', 'Length', 'Weight']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=46)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression(C=100, max_iter=1000, random_state=46, solver='lbfgs')
lr.fit(train_scaled, train_target)

print("train_set score" , lr.score(train_scaled,train_target))
print("test_set score" , lr.score(test_scaled,test_target))

print('분류 확률 :', lr.predict_proba(train_scaled[:5]))
print('로지스틱 회귀 계수 :', lr.coef_, lr.intercept_)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=2))


