import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 128
hm_epochs = 5

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder('float')

def conv_2D(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
	# strides taking uar sizes and move the window ..one pixwl at atime 
	# padding is used for following exapmles-> image is 28*28 and window size is 5 so for last window few pixels may not be available

def maxpool_2D(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# above						 size of window	  movement of window	
	# 2*2 pooling and move 2*2 pooling here

def convolutional_neural_network_model(x):
	weights = {'w_convl1':tf.Variable(tf.random_normal([5,5,1,32])),#5*5 convolution, take 1 input and will produce 32 features
				'w_convl2':tf.Variable(tf.random_normal([5,5,32,64])),
				'w_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
				'out':tf.Variable(tf.random_normal([1024, n_classes]))
				}

	biases = {
		'b_convl1':tf.Variable(tf.random_normal([32])),#5*5 convolution, take 1 input and will produce 32 features
		'b_convl2':tf.Variable(tf.random_normal([64])),
		'b_fc':tf.Variable(tf.random_normal([1024])),
		'out':tf.Variable(tf.random_normal([n_classes]))
			}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	# reshaping 184 pixel image to a flat 28*28 image
	
	conv1 = tf.nn.relu(conv_2D(x,weights['w_convl1'] + biases['b_convl1']))
	conv1 = maxpool_2D(conv1)

	conv2 = tf.nn.relu(conv_2D(conv1,weights['w_convl2'] + biases['b_convl2']))
	conv2 = maxpool_2D(conv2)
	
	fc = tf.reshape(conv2,[-1,7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])

	fc = tf.nn.dropout(fc,keep_rate)
	# here dropout is 0.8 i.e. 80% of the neurons passed further
	# for much larger dataset drop outs are used and impacts lot :)

	output = tf.matmul(fc, weights['out']) + biases['out']
	return output

def train_neural_network(x):
	prediction = convolutional_neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)