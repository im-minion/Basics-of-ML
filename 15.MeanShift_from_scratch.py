import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

centers = random.randrange(2,5)

X, y = make_blobs(n_samples=100, centers=centers,n_features=2)
style.use('ggplot')

# X = np.array([
# 	[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3]
# ])

# plt.scatter(X[:,0],X[:,1],s=150)
# plt.show()

colors = 10*["g","r","b","k","c","y"]

# Hierarchical Clustering

class MEANSHIFT_:
	def __init__(self,radius=None,radius_norm_step = 30):
		self.radius = radius
		self.radius_norm_step = radius_norm_step
	
	def fit(self,data):
		centroids ={}
		if self.radius == None:
			all_data_centroid = np.average(data,axis=0)
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm / self.radius_norm_step


		for i in range(len(data)):
			centroids[i] = data[i]
			# all datapoitns are cluster itself
		
		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]

				weights = [i for i in range(self.radius_norm_step)][::-1]


				for feature in data:
					distance = np.linalg.norm(feature-centroid)
					if distance == 0:
						distance = 0.000001
					weight_index = int(distance/self.radius)
					if weight_index > self.radius_norm_step-1:
						weight_index = self.radius_norm_step-1
					to_add = (weights[weight_index]*2)*[feature]
					in_bandwidth += to_add
					
					new_centroid = np.average(in_bandwidth,axis=0)
				new_centroids.append(tuple(new_centroid))
			
			uniques = sorted(list(set(new_centroids)))

			to_pop =[]
			for i in uniques:
				for ii in uniques:
					if i==ii:
						pass
					elif np.linalg.norm(np.array(i) - np.array(ii)) < self.radius:
						to_pop.append(ii)
						break

				for i in to_pop:
					try:
						uniques.remove(i)
					except:
						pass
			prev_centroids = dict(centroids)

			centroids = {}

			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])

			opimized = True
			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					opimized = False
				if not opimized:
					break
			if opimized:
				break
		
		self.centroids = centroids

		self.classifications = {}

		for i in range(len(self.centroids)):
			self.classifications[i] = []
		for feature in data:
			distances = [np.linalg.norm(feature-self.centroids[centroid])for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(feature)


	def predict(self,data):
		distances = [np.linalg.norm(feature-self.centroids[centroid])for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification
			
clf = MEANSHIFT_()
clf.fit(X)

centroids = clf.centroids
print(centroids)

for classification in clf.classifications:
	color = colors[classification]
	for feature in clf.classifications[classification]:
		plt.scatter(feature[0],feature[1],marker='o',color=color,s=150,linewidths=5)

plt.scatter(X[:,0],X[:,1],s=150)

for c in centroids:
	plt.scatter(centroids[c][0],centroids[c][1],color='k',marker='*',s =150)
plt.show()

