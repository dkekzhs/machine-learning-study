import numpy as np
import random
from scipy.spatial import distance
class KMeans:
    def __init__(self, n_clusters=3, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def new_center(self,clusters,X_train,n):
        sums = np.zeros((n, len(X_train[0])))
        a = {}
        for i in range(n):
            a[i] = 0
        for i,j in zip(clusters,X_train):
            a[i] +=1
            sums[i] += j

        for i in range(n):
            sums[i] = sums[i] / a[i]
        return sums
    def find_cluster(self,a):
        return a.index(min(a))
    def append_cluster(self,all):
        clusters= []
        for i in all:
            clusters.append(self.find_cluster(i))
        return clusters

    def center_distance(self,X_train,centers):
        all_dist = []
        tmp_dist = []
        for i in range(len(X_train)):
            for j in range(len(centers)):
                dist = np.linalg.norm(X_train[i] - centers[j])
                tmp_dist.append(dist)

            all_dist.append(tmp_dist)
            tmp_dist = []
        return all_dist
    def fit(self, X_train):
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = np.array([random.uniform(min_, max_) for _ in range(self.n_clusters)])

        # 초기화
        rand_point = random.sample(range(0, len(X_train)), self.n_clusters)
        self.centroids = []
        for i in rand_point:
            self.centroids.append(X_train[i])

        for i in range(self.max_iter):
            all = self.center_distance(X_train,self.centroids)
            self.clusters = self.append_cluster(all)
            self.centroids = self.new_center(self.clusters,X_train,self.n_clusters)
            print(i," : ",np.unique(self.labels_(),return_counts=True))


    def labels_(self):
        return self.clusters




fruits = np.load('fruits_300.npy')

fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA

pca = PCA(n_components = 50)
pca.fit(fruits_2d)


fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

km = KMeans(n_clusters=3, max_iter=10)
km.fit(fruits_pca)
print(np.unique(km.labels_(),return_counts=True))
