import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

dataA = "152 165 228 275 148 436 191 388 240 154 92 257 435 166 379 352 412 204 152 271 229 299 120 243 389 141 423 347 182 197 433 432 231 267 122 353 394 351 369 144 403 180 343 260 113 343 346 405 231 114 88 295 366 298 413 232 156 87 169 142 426 324 407 240 401 371 388 111 309 118 124 348 83 317 419 232 80 424 166 278 277 357 225 301 236 150 210 171 189 265 165 240 194 311 316 217 192 391 264 96"
data1 = []
for i in dataA.split(" "):
  data1.append([int(i)])
data1 = np.array(data1)

dataB="28 11 18 41 32 32 36 24 46 22 21 19 47 13 46 26 44 16 33 29 12 20 14 14 42 21 39 39 26 14 25 35 11 21 13 27 10 47 42 15 49 35 44 28 36 23 43 49 29 36 18 38 39 26 13 49 17 16 26 38 46 42 44 25 47 31 35 33 36 43 20 42 44 41 42 18 26 22 23 24 17 32 34 26 38 14 48 46 48 32 31 49 21 12 41 31 20 38 39 11"
data2 = []
for i in dataB.split(" "):
  data2.append([int(i)])
data2 = np.array(data2)

dataC="9227.12611099 15209.80474098 9311.95993067 21263.77343263 10528.75134887 10876.92474023 15342.00805349 8913.29421852 31492.31548856 9033.1660989 10260.50656699 12585.33156544 39834.35974033 15150.84066707 35704.77621388 10536.86287224 27597.33876426 16123.51834194 13058.1735733 6819.08264784 18551.32613415 13404.91982785 13222.11092249 13964.09382384 23582.8457269 7436.52813824 19706.50565295 18171.07662177 8413.29809825 12275.519076 12735.75639089 16067.50818052 16856.66710537 10204.40729877 17180.55361108 13126.83375061 15206.37769575 36581.3756277 23603.92422716 13250.52799835 43776.38553781 13541.97677532 26432.02413518 9918.23448976 12701.23602668 10830.50704778 26449.10264136 39348.64741535 10179.28303126 12843.76166282 12396.03136948 18677.67293277 16468.57308247 12729.64835531 15004.78217872 41145.2344778 14795.58509886 15573.17989138 11353.27364851 19476.79886177 31228.87453181 25510.92012528 30411.50156982 10181.39998825 36349.98147558 12055.15351343 13753.29077888 9544.71020363 14685.73191073 25549.03161232 10798.82390998 22849.79265161 25781.79434806 16514.1061051 25141.83111559 12337.45709393 10920.98090119 10923.3290181 14192.51296257 11564.82173848 13052.55434434 11236.10082548 12466.67627549 11420.93798492 19897.34338251 17903.61449945 41272.84428027 35346.99156573 36382.18822842 10754.75298271 15939.58962254 46316.50662613 12128.86949794 14359.37594983 25821.82349065 11862.47895739 15400.09535732 18269.82255117 18965.62459712 20996.40664895"
data3 = []
for i in dataC.split(" "):
  data3.append([float(i)])

data3 = np.array(data3)

# 데이터 2,3 데이터2 = 10~50
# plt.scatter(data2,data3)
# plt.show()

train_input, test_input, train_target, test_target = train_test_split(
data2, data3, random_state=42)
test_input = test_input.reshape(-1,1)
train_input = train_input.reshape(-1,1)

knr = KNeighborsRegressor()
knr.fit(train_input,train_target)
print("knrScore test score= ",knr.score(test_input,test_target))
print("KnrScore train score= ",knr.score(train_input,train_target))

lr = LinearRegression()
lr.fit(train_input, train_target)

## 다항 회귀
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)
point = np.arange(10,50)
plt.scatter(train_input,train_target)
print(lr.coef_,lr.intercept_)
plt.plot(point,53*point**2 - 2615*point + 41325)
y = lr.predict([[53**2,53]])
plt.scatter(53,y , marker="^")
plt.show()
print("lr score train = ",lr.score(train_poly, train_target))
print("lr score test = ",lr.score(test_poly, test_target))


