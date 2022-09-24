import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#도미 데이터
bream_length = np.random.randn(1,49)*3+30
bream_weight = bream_length*10+np.random.randn(1,49)*10

plt.figure(1)
plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#빙어 데이터
smelt_length= np.random.randn(1,21)*2+12
smelt_weight= smelt_length+np.random.randn(1,21)*2

plt.figure(2)
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length=np.concatenate((bream_length,smelt_length),axis=1)
weight=np.concatenate((bream_weight,smelt_weight),axis=1)

fish_data = np.concatenate((length,weight),axis=0).transpose()#입력
fish_target = [1]*49 + [0]*21#출력
# 입력과 mapping할 답이 마련되면 ==> Mapping 작업을 합니다.
kn = KNeighborsClassifier() #학습 알고리즘
kn.fit(fish_data, fish_target) #컴퓨터가 학습 Mapping
kn.score(fish_data, fish_target)
#kn은 물고기를 두가지로 구분하는 인공지능입니다.
plt.figure(3)
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(10, 50, marker='^')
plt.scatter(40, 300, marker='^')
plt.scatter(15, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print('결과는:', kn.predict([[10, 50]]))
print('결과는:', kn.predict([[40, 300]]))
print('결과는:', kn.predict([[15, 150]]))

#%%