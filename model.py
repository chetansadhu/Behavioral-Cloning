'''
    Project 3: Behaviour cloning
    Author: Chetan Sadhu
'''

# Import modules
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, Lambda, Cropping2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout
from keras.regularizers import l2, activity_l2
from keras import backend as K
import tensorflow as tf
from sklearn.utils import shuffle
import csv
from sklearn.model_selection import train_test_split

# Read the data from the csv file
samples = []
with open('data_run/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split the samples with 80-20 ratio for train-validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    '''
        Function to generate the batches for training and validation data
        samples[in] - Samples of the training set and validation set out of which batches has to be created
        batch_size[in] - Batch size of each batch of the training and validation set. Defaults to 32
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Read the images and convert to RGB color space
                center_image_name = 'data_run\\IMG\\' + batch_sample[0].split('\\')[-1]
                left_image_name = 'data_run\\IMG\\' + batch_sample[1].split('\\')[-1]
                right_image_name = 'data_run\\IMG\\' + batch_sample[2].split('\\')[-1]
                center_image = cv2.imread(center_image_name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.imread(left_image_name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.imread(right_image_name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)

                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                # 0.12 steering angle correction factor               
                angles.append(center_angle + 0.12)
                angles.append(center_angle - 0.12)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Get training and validation batches
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# Allow gpu memory to grow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)


## Network architecture
model = Sequential()
# Cropping layer. Crop the images to 90 x 320. Cuts 40 pixels at the top and 20 pixels at the bottom
model.add(Cropping2D(cropping=((40,20),(0,0)), input_shape=(160,320, 3)))
# Lambda layer for image normalization and zero centering
model.add(Lambda(lambda x: x/255. - 0.5))
# Convolution layer 3x3 kernel with 2x2 stride. input = (:, 90, 320, 3) output = (:, 44, 159, 6)
model.add(Conv2D(nb_filter=6, nb_row=3, nb_col=3, subsample=(2, 2), activation='relu'))
# Convolution layer 3x3 kernel with 2x2 stride. input = (:, 44, 159, 6) output = (:, 21, 79, 18)
model.add(Conv2D(nb_filter=18, nb_row=3, nb_col=3, subsample=(2, 2), activation='relu'))
# Convolution layer 3x3 kernel with 2x2 stride. input = (:, 21, 79, 18) output = (:, 11, 39, 36)
model.add(Conv2D(nb_filter=36, nb_row=3, nb_col=3, subsample=(2, 2), activation='relu'))
# Convolution layer 5x5 kernel with 1x1 stride. input = (:, 11, 39, 36) output = (:, 7, 35, 48)
model.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, activation='relu'))
# Convolution layer 5x5 kernel with 1x1 stride. input = (:, 7, 35, 48) output = (:, 3, 31, 64)
model.add(Conv2D(nb_filter=64, nb_row=5, nb_col=5, activation='relu'))
# Flatten - (:, 5952)
model.add(Flatten())
# Fully connected layer with 200 output node
model.add(Dense(200, activation='relu'))
# Fully connected layer with 75 output node
model.add(Dense(75, activation='relu'))
# Fully connected layer with 15 output node
model.add(Dense(15, activation='relu'))
# Fully connected layer with 1 output node
model.add(Dense(1))
# Compile with mean square error loss and adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=3*len(train_samples), validation_data=validation_generator, nb_epoch=7, nb_val_samples=3*len(validation_samples))
model.save('model.h5')