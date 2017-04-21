# -*- coding: utf-8 -*-
import re
import tensorflow as tf
import config
from PIL import Image
import numpy as np
import json
from sys import path
from shutil import copyfile
from glob import glob
from random import shuffle

char_dict = {}
number_dict = {}

current_index = 0
current_test_index = 0

total_img = []
total_test_img = []


# if the folder has the img
def move_img():
    i = 0
    with open(config.VALIDATE_DATA_JSON, 'r') as train:
        for p in train.readlines():
            i = i + 1
            hit = json.loads(p)
            read_img_and_move(hit, i)

    print 'done'


def read_img_and_move(hit, index):
    image_name = hit['hit_input'].split('/')[-1]
    image_val = hit['hit_result']['num'].upper()
    full_file_name = '/'.join([config.VALIDATE_IMG_PATH, image_name])
    # print full_file_name, image_val

    dst_file = '/'.join([config.VALIDATE_IMG_DST_PATH, str(index) + '_' + image_val + '.jpeg'])

    print 'handle the file of %s' % dst_file
    copyfile(full_file_name, dst_file)


def read_one_img(file_path):
    img = Image.open(file_path)
    img = img.convert("L")

    source = np.array(img)

    if source.shape != (53, 130):
        print 'find one shape is not right %s' % file_path
        return None, None

    img_data = source.flatten() / 255.0
    file_name = file_path.split('/')[-1]
    file_name = re.search('_(.{4})[.]', file_name)

    if not file_name:
        print 'since the result is None :%s' % file_path
        return None, None

    text = file_name.group(1)
    if '0' in text:
        text = text.replace('0', 'O')

    return text_to_array(text), img_data


# prepare the char to index
def prepare_char_dict():
    if char_dict:
        return char_dict

    for index, val in enumerate(config.VALIDATE_STRING):
        char_dict[val] = index

    return char_dict


def prepare_number_dict():
    if number_dict:
        return number_dict

    for index, val in enumerate(config.VALIDATE_STRING):
        number_dict[index] = val

    return number_dict


def text_to_array(text):
    char_dict_tmp = prepare_char_dict()

    arr = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN, dtype=np.int8)
    for i, p in enumerate(text):
        key_index = char_dict_tmp[p]
        index = i * config.CHAR_SET_LEN + key_index
        arr[index] = 1

    return arr


def array_to_text(arr):
    num_dict_tmp = prepare_number_dict()
    text = []
    char_pos = arr.nonzero()[0]
    for index, val in enumerate(char_pos):
        if index == 0:
            index = 1
        key_index = val % (index * config.CHAR_SET_LEN)
        text.append(num_dict_tmp[key_index])
    return ''.join(text)


def read_train_folder():
    res = []

    train_path = config.REAL_TRAIN_PATH + '/*.jpeg'
    for img in glob(train_path):
        res.append(img)

    print 'read once '
    shuffle(res)

    global total_img
    total_img = res


def read_test_folder():
    global total_test_img

    test_path = config.REAL_TEST_PATH + '/*.jpeg'
    if total_test_img:
        print 'already read one,please check it '
        return

    for img in glob(test_path):
        total_test_img.append(img)

    print 'read once with test folder'


def gen_train_batch(batch_size):
    global current_index

    # prepare the total img
    if not total_img:
        read_train_folder()

    start = current_index * batch_size
    end = (current_index + 1) * batch_size

    batch_x = np.zeros([batch_size, config.IMAGE_WIDTH * config.IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, config.MAX_CAPTCHA * config.CHAR_SET_LEN])

    i = 0
    for img in total_img[start:end]:
        label, image = read_one_img(img)

        if image is None:
            continue

        batch_y[i, :] = label
        batch_x[i, :] = image
        i = i + 1

    current_index = current_index + 1
    return batch_x, batch_y


def gen_test_batch(batch_size):
    global current_test_index

    if not total_test_img:
        read_test_folder()

    start = current_test_index * batch_size
    end = (current_test_index + 1) * batch_size

    batch_x = np.zeros([batch_size, config.IMAGE_WIDTH * config.IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, config.MAX_CAPTCHA * config.CHAR_SET_LEN])

    i = 0
    for img in total_test_img[start:end]:
        label, image = read_one_img(img)

        if image is None:
            continue

        batch_x[i, :] = image
        batch_y[i, :] = label
        i = i + 1

    current_test_index = current_test_index + 1
    return batch_x, batch_y


if __name__ == '__main__':
    # print gen_train_batch(64)
    # print gen_train_batch(64)
    print gen_test_batch(32)
    # print gen_test_batch(32)
    # print gen_test_batch(32)
    # print current_index
    print current_test_index
