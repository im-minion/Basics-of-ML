import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

X = np.array([
	[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]
	])

# plt.scatter(X[:,0],X[:,1],s=150)
# plt.show()
clf = KMeans(n_clusters=6)
# it is obvious that =>
# number of clusters must be equal to
# atleast number of data points you have

clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

color =["g.","r.","c.","b.","k.","y."]

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],color[labels[i]],markersize = 20)

plt.scatter(centroids[:,0],centroids[:,1], marker = 'x',s=150,linewidths=5)
plt.show()
print(centroids,labels)