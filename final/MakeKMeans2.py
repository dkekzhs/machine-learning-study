import numpy as np
import random

def center_init(n,X):
    # min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
    # centers = np.array([random.uniform(min_, max_) for _ in range(n)])
    # return centers
    rand_point = random.sample(range(0, len(X)), n)

    centers = []
    for i in rand_point:
        centers.append(X[i])
    return centers
def find_cluster(a):
    return a.index(min(a))

def new_center(clusters,X,n):
    sums = np.zeros((n, len(X[0])))

    for i,j in zip(clusters,X):
        sums[i] += j
    return sums / len(X)

def center_distance(X,centers):
    all_dist = []
    tmp_dist = []
    for i in range(len(X)):
        for j in range(len(centers)):
            dist = np.linalg.norm(centers[j]-X[i])

            tmp_dist.append(dist)
        all_dist.append(tmp_dist)
        tmp_dist = []
    return all_dist

def append_cluster(all_dist):
    clusters = []
    for i in all_dist:
        clusters.append(find_cluster(i))
    return clusters

def KMeans(n,X,iter_count):
    centers = center_init(n,X)
    for i in range(iter_count):
        all_dist = center_distance(X,centers)
        clusters = append_cluster(all_dist)
        centers = new_center(clusters,X,n)
    return clusters


fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA

pca = PCA(n_components = 50)
pca.fit(fruits_2d)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

a = KMeans(n=3,X=fruits_pca,iter_count=1)
print(np.unique(a,return_counts=True))
