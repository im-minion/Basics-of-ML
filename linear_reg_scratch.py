from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
# draw best fit line 

xs = np.asarray([1,2,3,4,5,6], dtype = np.float64)
ys = np.asarray([5,4,6,5,6,7], dtype = np.float64)

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

print(r_squarred)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='r') # put the prediction point
plt.plot(xs,regression_line)
plt.show()