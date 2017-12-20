import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
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

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)

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

df = handle_non_numeric_data(df)
# here we can try droping the ticket column and boat column
df.drop(['boat','ticket'],1, inplace=True)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = { }
n_clusters_ = len(np.unique(labels))
for i in range(n_clusters_):
	temp_df = original_df[(original_df['cluster_group'] == float(i))]
	survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
	survival_rate = len(survival_cluster) / len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)