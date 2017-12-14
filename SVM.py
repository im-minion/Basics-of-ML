# we have support vectors
# at T Bell labs, most popular 
# binary classifier
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

accuracies = []
for  i in range(25):
  df =  pd.read_csv('breast-cancer-wisconsin.data')
  df.replace('?',-99999, inplace=True)
  df.drop(['id'],1,inplace=True) 

  X = np.array(df.drop(['class'],1))
  y = np.array(df['class'])

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

  clf = svm.SVC()	#same preogram as KNN just replacing this line for chanaging algoritm
  clf.fit(X_train, y_train)

  accuracy = clf.score(X_test, y_test)

  print('Accuracy : ',accuracy)
  accuracies.append(accuracy)
  # example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,2,1,3,2,1]])
  # example_measures = example_measures.reshape(len(example_measures),-1)
  # prediction = clf.predict(example_measures)
  # print(prediction)

print('Accuracies : ',sum(accuracies)/len(accuracies))