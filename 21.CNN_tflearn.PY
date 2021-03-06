import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X,Y, test_x, test_y = mnist.load_data(one_hot=True)
# one hot -> one will active at aone time

X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

convnet = input_data(shape=[None,28,28,1], name='input')

# in case if u dont know , relu means rectified linear

# conv1 layer
convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)

# conv2 layer
convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)

# fully connected layer
convnet = fully_connected(convnet,1024,activation='relu')
convnet = dropout(convnet,0.8)

# output layer
convnet = fully_connected(convnet,10,activation='softmax')
convnet = regression(convnet,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy',name='targets')


model = tflearn.DNN(convnet)
# DNN -> deep neural network

'''
# uncomment this whole thing for the first tym running
model.fit(
	{'input':X}, 
	{'targets':Y}, 
	n_epoch=10,
	validation_set=(
		{'input':test_x},{'targets':test_y}
		),
	snapshot_step=500,
	show_metric=True,
	run_id='mnist'
	)

model.save('tflearncnn.model')
# this will save the model not the dataset and not the pickel type
# once saving is done and we are ready to use that 'tflearncnn.model' file as below
'''
model.load('tflearncnn.model')

# print( model.predict( [test_x[1]] ) )
# check above line few data losses error are there