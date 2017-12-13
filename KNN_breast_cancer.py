import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
# ecludein_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

def k_nearest_neighbors(data, predict, k=3):
	if(len(data) >= k):
		warnings.warn('k is set to value less than total voting groups');
	# knnalgo
	distances = []
	for group in data:
		for features in data[group]:
			# ecludein_distance = np.sqrt( np.sum( (np.array(feature) - np.array(predict)**2 ) ) )
			ecludein_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([ecludein_distance, group])
	votes = [i[1] for i in sorted(distances)[:k]]
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	
	# print(vote_result, confidence)

	return vote_result, confidence
accuracies = []
for i in range(25):
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?',-99999,inplace=True)
	df.drop(['id'],1,inplace=True)
	# print(df.head())
	full_data = df.astype(float).values.tolist()
	random.shuffle(full_data)
	test_size = 0.4

	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}

	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]
	# last 20% will be test data

	for i in train_data:
		train_set[i[-1]].append(i[:-1])

	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for gropu in test_set:
		for data in test_set[gropu]:
			vote, confidence = k_nearest_neighbors(train_set,data,k=5)
			if gropu == vote:
				correct +=1
			# else:
			# 	print(confidence)
			total +=1
	print('Accuracy :',correct/total)
	accuracies.append(correct/total)

print('Accuracies : ',sum(accuracies)/len(accuracies))