import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.3 + 0.1

Weights = tf.Variable(tf.random_uniform([1], -1, 1))
biases = tf.Variable(tf.zeros([1]))
y_predict = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y_predict - y_data))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init = tf.initialize_all_variables()

conf = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=conf) as sess:
    sess.run(init)
    for step in range(100):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(Weights), sess.run(biases)

