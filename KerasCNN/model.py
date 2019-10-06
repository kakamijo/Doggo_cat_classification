# from https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification
# RGB images classification

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, FAST_RUN, train_generator, total_validate, \
    total_train, batch_size, test_generator, nb_samples, test_df, validation_generator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

model = Sequential()
# Layer1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Layer2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Layer3
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Fully Connected Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 for cats and doggo
# configure the model for training
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# prints a summary representation of model
model.summary()

# apply callbacks(set of functions to be applied at the training procedure)
# EarlyStopping(stop training when a monitored quantity has stopped improving while "patience" epochs)
earlystop = EarlyStopping(monitor='val_loss', patience=10)
# ReduceLROnPlateau(reduce learning rate when a metric has stopped improving)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# FAST_RUN: True->3 epochs,False->50 epochs
epochs = 3 if FAST_RUN else 50

# train the model on data generated batch-by-batch
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks
)

# save the weights of the model
model.save_weights("model.h5")
# predict
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))

# pick the category that have the highest probability
test_df['category'] = np.argmax(predict, axis=-1)
# convert predict category -> generator classes
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
# dog:cat -> 1:0
test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

# show as bar
test_df['category'].value_counts().plot.bar()
plt.show()
