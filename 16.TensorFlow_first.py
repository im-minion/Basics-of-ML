import tensorflow as tf
# first
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# second program
x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)

# print(result)


# sess = tf.Session()
# print(sess.run(result))
# sess.close()

with tf.Session() as sess:
	output = sess.run(result)
	print(output)
# cant access output varible here
# Environment path:-
# export PATH="$PATH:/usr/local/cuda-8.0/bin" export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"