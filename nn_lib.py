# coding:utf8
import numpy as np
import tensorflow as tf


def print_me():
    print("Hello, data!")


def print_digit(data, labels, plt, index=None):
    if index is None:
        random_digit = np.random.randint(0, high=data.shape[1])
    else:
        random_digit = index
    pixels = data[random_digit].reshape((28, 28))
    label = np.where(labels[random_digit] == 1)[0][0]
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='Greys')
    plt.show()


def linear_regression_mnist(mnist, num_batches):
    loss = np.array([])
    sess = tf.InteractiveSession()
    # nodes for the input images and target output classes. None corresponding to the batch size
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float64, shape=[None, 10])
    # variables
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # linear regression model
    y = tf.matmul(x, W) + b
    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    # шаг обучения
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(num_batches):
        batch = mnist.train.next_batch(100)
        _, current_loss = sess.run(
            [train_step, cross_entropy],
            feed_dict={x: batch[0].astype(np.float32),
                       y_: batch[1].astype(np.float32)})
        loss = np.append(loss, current_loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def deep_mnist(mnist, num_batches):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    # convolution layer: patch size 5x5,1 input channel, 32 output dimension
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 1 - num of color channel
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # convolution and max pooling
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 2nd convolution layer - 64 features for each 5x5 patch.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # fullu connected
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # layer softmax regression
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # final layes
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # model expluatation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_batches):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print("interactions finished!")
        print(
            'test accuracy %g' % accuracy.eval(
                feed_dict={x: mnist.test.images[:6000], y_: mnist.test.labels[:6000], keep_prob: 1.0})
        )
