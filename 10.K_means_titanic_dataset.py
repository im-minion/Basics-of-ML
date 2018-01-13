import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing

style.use('ggplot')

'''
pclass
survived
name
sex
age
sibsp
parch
ticket
fare
cabin
embarked	
boat
body
home.dest

predict servived or not?

'''


# print(df.head())

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


df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
# here we can try droping the ticket column and boat column

df.drop(['boat','ticket'],1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
df = handle_non_numeric_data(df)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predict_this = np.array(X[i].astype(float))
	predict_this = predict_this.reshape(-1,len(predict_this))
	prediction = clf.predict(predict_this)
	if prediction[0] == y[i]:
		correct += 1

print(correct/len(X))