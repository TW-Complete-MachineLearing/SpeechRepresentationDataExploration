import tensorflow as tf


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


class cnn_model(object):
    def __init__(self, conv1_weight_shape,
                 conv1_bias_shape,
                 conv2_weight_shape,
                 conv2_bias_shape,
                 nn_weight_shape,
                 nn_bias_shape,
                 input_shape,
                 output_weight_shape,
                 output_bias_shape):
        self.output_bias_shape = output_bias_shape
        self.output_weight_shape = output_weight_shape
        self.nn_bias_shape = nn_bias_shape
        self.nn_weight_shape = nn_weight_shape
        self.input_shape = input_shape
        self.conv2_bias_shape = conv2_bias_shape
        self.conv1_bias_shape = conv1_bias_shape
        self.conv2_weight_shape = conv2_weight_shape
        self.conv1_weight_shape = conv1_weight_shape
        self.__init_model()

    def __init_model(self):
        self.W_conv1 = init_weights(self.conv1_weight_shape)
        self.b_conv1 = bias_variable(self.conv1_bias_shape, 0.1)
        self.W_conv2 = init_weights(self.conv2_weight_shape)
        self.b_conv2 = bias_variable(self.conv2_bias_shape, 0.1)
        self.W_nn1 = init_weights(self.nn_weight_shape)
        self.b_nn1 = bias_variable(self.nn_bias_shape, 0.1)
        self.W_output = init_weights(self.output_weight_shape)
        self.b_output = bias_variable(self.output_bias_shape)

    def get_output(self, sample):
        x_image = tf.reshape(sample, self.input_shape)
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_flatten = tf.reshape(h_pool2, [-1, self.nn_weight_shape[0]])
        h_nn1 = tf.nn.relu(tf.matmul(h_pool2_flatten, self.W_nn1) + self.b_nn1)

        keep_prob = tf.placeholder(tf.float32)
        h_nn1_drop = tf.nn.dropout(h_nn1, keep_prob)

        output = tf.nn.softmax(tf.matmul(h_nn1_drop, self.W_output) + self.b_output)
        return output
