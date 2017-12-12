from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')
# draw best fit line 



#xs = np.asarray([1,2,3,4,5,6], dtype = np.float64)
#ys = np.asarray([5,4,6,5,6,7], dtype = np.float64)

###########
# following lines to create random dataset

def create_dataset(hm, varience, step = 2,correlation = False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-varience,varience)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype=np.float64), np.array(ys, dtype=np.float64)

###############
def best_fit_slope_intercept(xs,ys):
	m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)) )
	# PEMDAS operations

	b = mean(ys) - (m * mean(xs))
	
	return m,b

def squared_error(ys_original, ys_line):
	return sum( (ys_line - ys_original)**2 )

def coefficeint_of_determination(ys_original, ys_line):
	y_mean_line = [mean(ys_original) for y in ys_original]
	squared_error_regression = squared_error(ys_original,ys_line)
	squared_error_y_mean = squared_error(ys_original,y_mean_line)
	return (1-(squared_error_regression / squared_error_y_mean))


# get the random dataset
xs ,ys = create_dataset(40,20,2,correlation='neg')


m, b = best_fit_slope_intercept(xs,ys)
# print(m,b)

regression_line = [(m*x) + b for x in xs]
# above line is similar to using for loop as ->
# for x in xs:
# 	regression_line.append((m*x)+b)


# for any x prediction can be given as -> 
predict_x = 8
predict_y = (m*predict_x) + b

r_squarred = coefficeint_of_determination(ys,regression_line)

print("\nR-Squared accuracy : ",r_squarred)

plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.scatter(predict_x,predict_y,color='r') # put the prediction point

plt.show()