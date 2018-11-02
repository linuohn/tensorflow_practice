import tensorflow as tf
import numpy as np

#%matplotlib inline # for ipyton interactive
%matplotlib notebook # for jupyter
import matplotlib.pyplot as plot
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='3'

## matrix to vector/ matrix
rand_matrix = np.random.rand(4,3) #size=(4,3), tf.Variable(tf.random_uniform([2,3], -1,1))
row_vector = rand_matrix[0]
row_matrix = rand_matrix[[0,1]]
column_to_row_vector = rand_matrix[:,1]
column_vector = rand_matrix[:,[1]]

### build nn & train
x_data = np.linspace(-1,1,500)[:,None].astype(np.float32)
noise = np.random.normal(0,0.1,[500,1]).astype(np.float32)
y_data = np.square(x_data) - 0.2 + noise
x_train, y_train = x_data[range(0,399), :], y_data[range(0,399), :]
x_test, y_test = x_data[range(399,500), :], y_data[range(399,400), :]

# build flow graph
def add_layer(input_data, input_size, output_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size], 0, 0.5))
    biases = tf.Variable(tf.zeros([output_size]) + 0.1)
    Wx_plus_b = tf.matmul(input_data, Weights) + biases

    if activation_function:
        return activation_function(Wx_plus_b)
    else:
        return Wx_plus_b

x_data_p = tf.placeholder(tf.float32, [None, 1])
y_data_p = tf.placeholder(tf.float32, [None, 1])

layer1_output = add_layer(x_data_p, 1, 10, tf.nn.relu)
predict = add_layer(layer1_output, 10, 1)
loss = tf.reduce_mean(tf.square(predict - y_data_p))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, \
                            allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options, \
                      allow_soft_placement=True,\
                      log_device_placement=False)
with tf.Session(config=config) as sess:
    fig = plot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_train, y_train)
    plt.ion() # interactive model
    plt.show()
    fig.canvas.draw() # update canvas

    sess.run(init)
    for step in range(10000):
        sess.run(train, feed_dict={x_data_p:x_train, y_data_p:y_train})
        if step % 1000 == 0:
            """print step, "train_loss:", \
                sess.run(loss, feed_dict={x_data_p:x_train, y_data_p:y_train}), \
                "test_loss:", \
                sess.run(loss, feed_dict={x_data_p:x_test, y_data_p:y_test}), \
                "predict:", \
                sess.run(predict, feed_dict={x_data_p:x_test}) # y_test is not necessary
            """
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction = sess.run(predict, feed_dict={x_data_p:x_test})
            ax.plot(x_test, prediction, 'r-')
            fig.canvas.draw()
            plt.pause(0.5)

