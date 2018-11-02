import tensorflow as tf
import numpy as np

#%matplotlib notebook # for jupyter
import matplotlib.pyplot as plot

### build nn & train
x_data = np.linspace(-1,1,500)[:,None].astype(np.float32)
noise = np.random.normal(0,0.1,[500,1]).astype(np.float32)
y_data = np.square(x_data) - 0.2 + noise
x_train, y_train = x_data[range(0,399), :], y_data[range(0,399), :]
x_test, y_test = x_data[range(399,500), :], y_data[range(399,400), :]

#test data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

# build flow graph
def add_layer(input_data, input_size, output_size, name, activation_function=None):
    with tf.variable_scope(name):
        Weights = tf.Variable(tf.random_normal([input_size, output_size], 0, 0.5), name='W')
        biases = tf.Variable(tf.zeros([output_size]) + 0.1, name='b')
        Wx_plus_b = tf.matmul(input_data, Weights) + biases

        if activation_function:
            return activation_function(Wx_plus_b)
        else:
            return Wx_plus_b

with tf.variable_scope('Input'):
    x_data_p = tf.placeholder(tf.float32, [None, 1], name="x_in")
    y_data_p = tf.placeholder(tf.float32, [None, 1], name="y_in")

with tf.variable_scope('NN'):
    layer1_output = add_layer(x_data_p, 1, 10, "layer1", tf.nn.relu)
    predict = add_layer(layer1_output, 10, 1, "layer2")

with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.square(predict - y_data_p))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, \
                            allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options, \
                      allow_soft_placement=True,\
                      log_device_placement=False)
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("./log", sess.graph)
    merge_op = tf.summary.merge_all()
    #fig = plot.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(x_train, y_train)
    #plt.ion() # interactive model
    #plt.show()
    #fig.canvas.draw() # update canvas

    sess.run(init)
    for step in range(10000):
        _, result = sess.run([train, merge_op], {x_data_p:x, y_data_p:y})
        if step % 1000 == 0:
            writer.add_summary(result, step)
            continue
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction = sess.run(predict, feed_dict={x_data_p:x_test})
            #ax.plot(x_test, prediction, 'r-')
            #fig.canvas.draw()
            #plt.pause(0.5)
