import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://bit.ly/perch_full.data')
perch_full = df.to_numpy()
#print(perch_full)
#print(np.shape(perch_full))

perch_weight = np.array(

    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,

     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,

     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,

     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,

     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,

     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,

     1000.0, 1000.0]

     )

target = train_test_split(perch_full, perch_weight, random_stste=42)
#에포크와 과대 과소 적합

import numpy as np

sc = SGDClassifier(loss='log', random_state=42)

train_score = []
test_score = []

classes = np.unique(train_target)

for _ in range(0, 300):
 sc.partial_fit(train_scaled, train_target, classes=classes)

 train_score.append(sc.score(train_scaled, train_target))
 test_score.append(sc.score(test_scaled, test_target))

 import matplotlib.pyplot as plt

 plt.plot(train_score)
 plt.plot(test_score)
 plt.xlabel('epoch')
 plt.ylabel('accuracy')
 plt.show()

 sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
 sc.fit(train_scaled, train_target)

 print(sc.score(train_scaled, train_target))
 print(sc.score(test_scaled, test_target))

 sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
 sc.fit(train_scaled, train_target)

 print(sc.score(train_scaled, train_target))
 print(sc.score(test_scaled, test_target))