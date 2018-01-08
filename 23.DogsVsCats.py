import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/vaibhav/Work/train'
TEST_DIR = '/home/vaibhav/Work/test'
IMG_SIZE = 50 # 50x50
# all images are not of same size so make them 
LR = 1e-3 # learning rate

MODEL_NAME = 'dogsvscat-{}-{}.model'.format(LR,'2con-basic-video')

def label_img(img):
	word_label = img.split('.')[-3]
	# now the word label will contain the word 'cat' or 'dog'

	if word_label == 'cat': return [1,0]
	elif word_label == 'dog': return [0,1]

def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path = os.path.join(TRAIN_DIR, img)
		img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE) )
		training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	np.save('traindata.npy',training_data)
	return training_data

def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR, img)
		img_num = img.split(',')[0] # id or the image file name
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,IMG_SIZE) )
		testing_data.append([np.array(img),img_num])
	np.save('test_data.npy',testing_data)
	return testing_data


# train_data = create_train_data()
# for the first tym this will create the 'traindata.npy' file
# after that comment above line and uncomment the following line
train_data = np.load('traindata.npy')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir="/tmp/tflearn_logs/")
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape( -1, IMG_SIZE, IMG_SIZE, 1 )
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape( -1, IMG_SIZE, IMG_SIZE, 1 )
test_y = [i[1] for i in test]

model.fit(
	{'input': X},
	{'targets': Y},
	n_epoch=5,
	validation_set=(
		{'input': test_x},
		{'targets': test_y}
		),
	snapshot_step=500,
	show_metric=True,
	run_id = MODEL_NAME
	)

# tensorboard --logdir = /tmp/tflearn_logs/
# /tmp/tflearn_logs/dogsvscat-0.001-2con-basic-video.model
