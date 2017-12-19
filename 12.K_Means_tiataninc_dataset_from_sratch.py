import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)

def handle_non_numeric_data(df):
	columns = df.columns.values
	# 'columns' will be list 
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1
			df[column] = list(map(convert_to_int,df[column]))
	return df

class KMEANS:
	def __init__(self,k=2,tol=0.001,max_iter=300):
		# tol = tolernace -> how much the centroid is gonna move
		# atmost how many times we wana run
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self,data):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i]
		
		for i in range(self.max_iter):
			self.classifications = {}
			for i in range(self.k):
				self.classifications[i] =[]

			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)
			
			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			
			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
					print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
					optimized = False

			if optimized:
				break

	def predict(self,data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification


colors =10*["g","r","c","b","k"]


df = handle_non_numeric_data(df)
# here we can try droping the ticket column and boat column
df.drop(['boat','ticket'],1, inplace=True)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = KMEANS()
clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1], marker="o", color="k",s=150,linewidths=5)

for classification in clf.classifications:
	c = colors[classification]
	# print(c)
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0],featureset[1], marker="x" , color = c,s=150,linewidths=5)


# correct = 0
# for i in range(len(X)):
# 	predict_this = np.array(X[i].astype(float))
# 	predict_this = predict_this.reshape(-1,len(predict_this))
# 	prediction = clf.predict(predict_this)
# 	if prediction[0] == y[i]:
# 		correct += 1

# print(correct/len(X))
