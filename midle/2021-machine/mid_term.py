import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from scipy.special import softmax

fish = pd.read_csv('https://bit.ly/fish_csv_data')
#fish_input = fish[:5].to_numpy()
fish_input = fish[['Height', 'Width', 'Length', 'Diagonal']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=45)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sc = SGDClassifier(loss='log', max_iter=1000, random_state=45)
sc.fit(train_scaled, train_target)

#print(sc.score(train_scaled, train_target))
#print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)

#print(sc.score(train_scaled, train_target))
#print(sc.score(test_scaled, test_target))
#print(train_input.shape, test_input.shape)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

#print(lr.score(train_scaled, train_target))
#(lr.score(test_scaled, test_target))
print('분류 예측 확률 -> ', lr.predict_proba(train_scaled[:5]))
print('로지스틱 회귀 계수 -> ', lr.coef_, lr.intercept_)

decision = lr.decision_fucntion(test_scaled[:5])
print(np.round(decision, decimals=2))
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=2))
