import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

bream_data_size = 10 # 도미
smelt_data_size = 20 # 빙어

bream_length = np.random.randn(1,bream_data_size)*5+35 # 평균 0 표준편차 1
bream_weight = bream_length*20+np.random.randn(1, bream_data_size)*20
# print(bream_length)
smelt_length= np.random.randn(1, smelt_data_size)*2+12
smelt_weight= smelt_length+np.random.randn(1, smelt_data_size)*2
'''
plt.figure(1)
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
'''
length=np.concatenate((bream_length,smelt_length),axis=1)
weight=np.concatenate((bream_weight,smelt_weight),axis=1)
# print(np.shape(length))

fish_data = np.concatenate((length,weight),axis=0).transpose() # 입력
# print(np.shape(fish_data))
fish_target = np.array([1]*bream_data_size + [0]*smelt_data_size) # 출력

np.random.seed(42)
index = np.arange(bream_data_size + smelt_data_size)
np.random.shuffle(index) # 마구 섞임
# print(index)

train_input = fish_data[index[0:60]] # 학습용 데이터
train_output = fish_target[index[0:60]]
# 학습 데이터로 학습을 시킨 모델에 학습 데이터로 테스트 하지 말 것 XXXX
test_input = fish_data[index[60:]] # 테스트 데이터
test_output = fish_target[index[60:]]

kn = KNeighborsClassifier() #학습 알고리즘
kn.fit(train_input, train_output) #컴퓨터가 학습 Mapping
# print(kn.score(test_input, test_output))

plt.figure(3)
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(25, 250, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
