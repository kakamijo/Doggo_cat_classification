import numpy as np
from keras import layers, optimizers, Input
from keras.layers import Dropout, Dense, Flatten
from preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, train_generator, total_validate, \
    total_train, batch_size, test_generator, nb_samples, test_df, validation_generator
from keras.models import Model
from keras.applications import ResNet50
import matplotlib.pyplot as plt


input_shape = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

# ResNet(include_top, weights, input_tensor, input_shape)
pre_trained_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_shape)

# search layer 'activation_49"
last_layer = pre_trained_model.get_layer('activation_49')
# get list of output tensor
last_output = last_layer.output

# flatten the output layer to 1 dimension
x = Flatten()(last_output)
# add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# add a dropout rate of 0.5
x = Dropout(0.5)(x)
# add a final sigmoid layer for classification
x = layers.Dense(2, activation='sigmoid')(x)

# VGG16 + added layers
model = Model(pre_trained_model.input, x)

# configure the model for training
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
# prints a summary representation of model
model.summary()

epochs = 20

# train the model on data generated batch-by-batch
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
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
