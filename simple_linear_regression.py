import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = "1xfQzjvZyaQZkyHgdUm3"
df = quandl.get('WIKI/GOOGL') 

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Open']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True) # preprocessing

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)
# print(df.head())

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

# x = x[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(x),len(y))
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.2)

# clf = svm.SVR(kernel='poly')
# clf = LinearRegression(n_jobs=-1)
# will parallely run as many jobs as posiible if -1 
#or mention the number of jobs should be executed int prallel


clf = LinearRegression()
clf.fit(x_train, y_train)
# purpose of saving classfier is to avoid the training step

###using pickle

with open('linearregression.pickle', 'wb') as f:
	pickle.dump(clf, f)
###this will store the trainng part in linearregression.pickle this file

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(x_test,y_test)

# print(accuracy)
forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()