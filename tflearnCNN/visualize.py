# from https://www.kaggle.com/sentdex/full-classification-example-with-convnet
# gray scale classification

import numpy as np
import matplotlib.pyplot as plt
from cnn2 import model
from preprocess import IMG_SIZE, process_test_data
from tqdm import tqdm

test_data = process_test_data()

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # cat: [1,0]
    # doggo:[0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()