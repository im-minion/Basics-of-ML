import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
# ecludein_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_feature = [5,7]

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
	print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]

	return vote_result

result = k_nearest_neighbors(dataset, new_feature, k=3)
print(result)

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0],new_feature[1], color=result)
plt.show()