# from https://www.kaggle.com/sentdex/full-classification-example-with-convnet
# gray scale classification

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '../input/train/train'  # set directly
TEST_DIR = '../input/test1/test1'
IMG_SIZE = 50
LR = 1E-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[0]  # ???->[-3]

    if word_label == 'cat':
        return [1, 0]  # [cat, doggo]
    elif word_label == 'dog':
        return [0, 1]


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data


def create_train_data():
    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()
