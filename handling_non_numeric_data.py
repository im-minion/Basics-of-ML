import numpy as np
import pandas as pd

df = pd.read_excel('titanic.xls')
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
print(df.head())

