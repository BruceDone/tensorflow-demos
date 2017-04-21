# -*- coding: utf-8 -*-
from train_model import create_layer
import tensorflow as tf
import config
import numpy as np
from gen_data import array_to_text, read_one_img


def crack_captcha(captcha_image):
    x_input = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT * config.IMAGE_WIDTH])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    output = create_layer(x_input, keep_prob)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN]), 2)

        print 'hello:',predict
        text_list = sess.run(predict, feed_dict={x_input: [captcha_image], keep_prob: 1})
        print text_list

        text = text_list[0].tolist()
        vector = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * config.CHAR_SET_LEN + n] = 1
            i += 1

        return array_to_text(vector)


def validate_image():
    text, image = read_one_img('/Users/brucedone/Data/weixin_captcha/test/3571_RCBE.jpeg')

    image = image.flatten() / 255.0
    predict_text = crack_captcha(image)
    print("label is : {} <----> predict is : {}".format(array_to_text(text), predict_text))


if __name__ == '__main__':
    validate_image()
