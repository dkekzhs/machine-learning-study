import numpy as np

from sklearn.cluster import KMeans



fruits = np.load('fruits_300.npy')

fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA

pca = PCA(n_components = 50)
pca.fit(fruits_2d)


fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

km = KMeans(n_clusters=3, max_iter=10)
km.fit(fruits_pca)
print(np.unique(km.labels_,return_counts=True))
