# -*- coding: utf-8 -*-

NUMBER = '0123456789'
CHAR_SMALL = 'abcdefghijklmnopqrstuvwxyz'
CHAR_BIG = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

MAX_CAPTCHA = 4  # 测试1位的训练准确率
VALIDATE_STRING = CHAR_BIG
CHAR_SET_LEN = len(VALIDATE_STRING)

IMAGE_HEIGHT = 53
IMAGE_WIDTH = 130

MAX_ACCURACY = 0.5

TRAIN_DATA_JSON = '/Users/brucedone/Data/weixin/228_run.json'
TRAIN_IMG_PATH = '/Users/brucedone/Data/yanzenma_20w/train_captcha'

VALIDATE_DATA_JSON = '/Users/brucedone/Data/weixin/226_run.json'
VALIDATE_IMG_PATH = '/Users/brucedone/Data/yanzenma_20w/test_captcha'

TRAIN_IMG_DST_PATH = '/data/weixin_captcha/train'
VALIDATE_IMG_DST_PATH = '/data/weixin_captcha/test'

REAL_TRAIN_PATH = '/data/weixin_captcha/train'
REAL_TEST_PATH = '/data/weixin_captcha/test'
