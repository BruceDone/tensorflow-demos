# -*- coding: utf-8 -*-
import tensorflow as tf
from gen_data import text_to_array
from config import MAX_CAPTCHA, CHAR_SET_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_ACCURACY
from gen_data import gen_train_batch, gen_test_batch

x_input = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])  # use the gray
y_input = tf.placeholder(tf.float32, [None, CHAR_SET_LEN * MAX_CAPTCHA])
keep_prob = tf.placeholder(tf.float32)


def __weight_variable(shape, stddev=0.01):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def __bias_variable(shape, stddev=0.1):
    initial = tf.random_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial)


def __conv2d(x, w):
    # strides 代表移动的平长
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def __max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_layer(x_input, keep_prob):
    x_image = tf.reshape(x_input, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    w_stddev = 0.1

    # 定义第1个卷积层
    w_c1 = __weight_variable([7, 7, 1, 32], stddev=w_stddev)  # 3x3 第一层32个卷积核 采用黑白色
    b_c1 = __bias_variable([32], stddev=0.1)
    h_c1 = tf.nn.relu(tf.nn.bias_add(__conv2d(x_image, w_c1), b_c1))  # 定义第一个卷积层
    h_pool1 = __max_pool_2x2(h_c1)  # 定义第一个池化层

    # 定义第2个卷积层
    w_c2 = __weight_variable([7, 7, 32, 64], stddev=w_stddev)
    b_c2 = __bias_variable([64], stddev=0.1)
    h_c2 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool1, w_c2), b_c2))
    h_pool2 = __max_pool_2x2(h_c2)

    # 定义第3个卷积层
    w_c3 = __weight_variable([7, 7, 64, 128], stddev=w_stddev)
    b_c3 = __bias_variable([128], stddev=0.1)
    h_c3 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool2, w_c3), b_c3))
    h_pool3 = __max_pool_2x2(h_c3)

    # print h_pool4
    w_c4 = __weight_variable([7, 7, 128, 128], stddev=w_stddev)
    b_c4 = __bias_variable([128], stddev=0.1)
    h_c4 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool3, w_c4), b_c4))

    # 全链接层1
    w_fc1 = __weight_variable([7 * 17 * 128, 2048], stddev=w_stddev)
    b_fc1 = __bias_variable([2048])

    h_pool3_flat = tf.reshape(h_c4, [-1, w_fc1.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, w_fc1), b_fc1))
    # drop out 内容0
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # 全链接层2
    w_output = __weight_variable([2048, MAX_CAPTCHA * CHAR_SET_LEN], stddev=w_stddev)
    b_output = __bias_variable([MAX_CAPTCHA * CHAR_SET_LEN])
    y_output = tf.add(tf.matmul(h_fc1_dropout, w_output), b_output)

    return y_output


def create_loss(layer, y_input):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=layer))
    return loss


def create_accuracy(output, y_input):
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    print 'predict', predict
    max_idx_p = tf.argmax(predict, 2)

    print 'max_idx_p', max_idx_p
    max_idx_l = tf.argmax(tf.reshape(y_input, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    print 'max_idx_p', max_idx_l
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def train():
    # create the layer and loss
    layer_output = create_layer(x_input, keep_prob)
    loss = create_loss(layer_output, y_input)
    accuracy = create_accuracy(layer_output, y_input)

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # save model
    saver = tf.train.Saver()

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        acc = 0.0
        i = 0

        while True:
            i = i + 1
            batch_x, batch_y = gen_train_batch(64)
            _, _loss = sess.run([train_step, loss],
                                feed_dict={x_input: batch_x, y_input: batch_y, keep_prob: 0.75})

            print(i, _loss)

            # 每100 step计算一次准确率
            if i % 50 == 0:
                batch_x_test, batch_y_test = gen_test_batch(100)
                acc = sess.run(accuracy, feed_dict={x_input: batch_x_test, y_input: batch_y_test, keep_prob: 1.})
                print('step is %s' % i, 'and accy is %s' % acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > MAX_ACCURACY or i == 650:
                    print('current acc > %s  ,stop now' % MAX_ACCURACY)
                    saver.save(sess, "weixin.model", global_step=i)
                    break


if __name__ == '__main__':
    train()
