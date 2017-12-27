import tensorflow as tf
session = tf.Session()

conv1_weight_shape = []
conv1_bias_shape = []
conv2_weight_shape = []
conv2_bias_shape = []
nn_weight_shape = []
nn_bias_shape = []
output_weight_shape = []
output_bias_shape = []
input_shape = []

def init_weights(shape):
    initial = tf.truncated_normal_initializer(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, bias_initial_value):
    initial = tf.constant(bias_initial_value, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides, padding):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def max_pool_2x2(x, ksize, strides, padding):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

W_conv1 = init_weights(conv1_weight_shape)
b_conv1 = bias_variable(conv1_bias_shape, 0.1)
W_conv2 = init_weights(conv2_weight_shape)
b_conv2 = bias_variable(conv2_bias_shape, 0.1)
W_nn1 = init_weights(nn_weight_shape)
b_nn1 = bias_variable(nn_bias_shape, 0.1)
W_output = init_weights(output_weight_shape)
b_output = bias_variable(output_bias_shape)

sample = tf.placeholder()
label = tf.placeholder()
x_image = tf.reshape(sample, input_shape)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flatten = tf.reshape(h_pool2, [-1, nn_weight_shape[0]])
h_nn1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_nn1) + b_nn1)

keep_prob = tf.placeholder(tf.float32)
h_nn1_drop = tf.nn.dropout(h_nn1, keep_prob)

output = tf.nn.softmax(tf.matmul(h_nn1_drop, W_output) + b_output)

cross_entropy = -tf.reduce_sum(label * tf.log(output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
cross_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

if __name__== '__main__':
    session.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print "test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

