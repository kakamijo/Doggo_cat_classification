# from https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification
# RGB images classification

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# labeling doggo and cats from a image name
filenames = os.listdir("../input/train/train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)  # doggo:1 / cat:0
    else:
        categories.append(0)

# Creating DataFrame(2-dimensional)
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})   # 0,1 -> cat,doggo (column data into string)

# percentage of train data, validation data -> 80 , 20 %
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)   # from The Hitchhiker's Guide(lmfao)
train_df = train_df.reset_index(drop=True)      # reset index
validate_df = validate_df.reset_index(drop=True)

# get the number of data and set batch size
total_train = train_df.shape[0]     # pandas.DataFrame.shape -> (row, column)
total_validate = validate_df.shape[0]
batch_size = 15

# augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# pandas DataFrame + path -> generates augmented/normalized data
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "../input/train/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,     # 128x128
    class_mode='categorical',
    batch_size=batch_size       # 15
)

# validation data augmentation / generate augmented data
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "../input/train/train/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# Prepare testing data
test_filenames = os.listdir("../input/test1/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "../input/test1/test1",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
