from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

df = pd.read_csv('finaldataset.csv')
xf = df[['keyword','grammar','qst']]
# intigrate keyword, grammar, qst :)
''''
keywords and qst:
e = 1
vg = 2
g = 3
o = 4
p = 5
vp = 6

Grammar:
y = 1
n = 0

class labels 0.1 to 0.9 simplifies to 0 to 9 for calculation purpose
'''

x = np.array(xf.values)
yf = df[['class']]
y = np.array(yf.values).ravel()
clf = GaussianNB()
clf.fit(x,y)

# predict for 
# 1. keyword = verygood, grammar = no, qst = ok 
# => 6
# 2. keyword = verygood, grammar = no, qst = vg 
# => 8
predicted = clf.predict([[2,0,4],[2,0,2]])

print(predicted)

# print(yf.values)
# x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
# y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
# # lx= np.array(
# # 	[
# # 		['e','y','e'],
# # 		['e','n','e'],
# # 		['vg','n','e'],
# # 		['e','y','vg'],
# # 		['e','n','g']
# # 	]
# # 	)
# # ly = np.array(
# # 	[
# # 		0.9,
# # 		0.9,
# # 		0.8,
# # 		0.8,
# # 		0.7
# # 	]
# # 	)
# print(yf.values)
# x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
# y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
# # lx= np.array(
# # 	[
# # 		['e','y','e'],
# # 		['e','n','e'],
# # 		['vg','n','e'],
# # 		['e','y','vg'],
# # 		['e','n','g']
# # 	]
# # 	)
# # ly = np.array(
# # 	[
# # 		0.9,
# # 		0.9,
# # 		0.8,
# # 		0.8,
# # 		0.7
# # 	]
# # 	)
# model = GaussianNB()

# # model.fit(lx,ly)
# model.fit(x,y)

# predicted = model.predict([[-1,-3],[-3,2]])
# print(predicted)