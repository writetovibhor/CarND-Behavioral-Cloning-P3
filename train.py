import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from keras.models import load_model
import glob
from math import exp, fabs
import random
from sklearn.model_selection import train_test_split
import sklearn
import keras
import time

samples = []
BATCH_SIZE = 64

with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("train_samples = {}, validation_samples = {}".format(len(train_samples), len(validation_samples)))

CORRECTION = [0.0, 0.25, -0.25]
MAX_OFFSET = 25.0

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def train_generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                idx = np.random.randint(3)
                base_angle = float(batch_sample[3]) + CORRECTION[idx]
                loop = True
                while loop:
                    loop = False
                    angle = base_angle

                    do_flip = (random.random() > 0.5)

                    if do_flip:
                        angle = -angle

                    x_offset = random.uniform(-MAX_OFFSET,MAX_OFFSET)
                    y_offset = random.uniform(-MAX_OFFSET,MAX_OFFSET)
                    angle = angle + (x_offset / MAX_OFFSET) * 0.5

                    if angle < 0.1 and random.random() < 0.2:
                        loop = True
                        continue

                    name = 'data/IMG/'+batch_sample[idx].split('/')[-1]
                    image = cv2.imread(name)
                    image = image[70:135, 0:320]

                    image = augment_brightness_camera_images(image)
                    img = add_random_shadow(image)

                    if do_flip:
                        image = cv2.flip(image,1)

                    image = auto_canny(image)
                    #out = np.random.randint(100)
                    #cv2.imwrite('temp/{}.png'.format(out),image)
                    #image = cv2.Canny(image,200,400)
                    #cv2.imwrite('temp/{}-canny.png'.format(out),image)
                    #print("temp/{}.png".format(out))

                    M = np.float32([[1,0,x_offset],[0,1,y_offset]])
                    rows = image.shape[0]
                    cols = image.shape[1]
                    image = cv2.warpAffine(image,M,(cols,rows))

                    image = cv2.resize(image,(64,64))
                    image = image[:,:,np.newaxis]

                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

def validation_generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                # # angle = float(batch_sample[3])
                image = image[70:135, 0:320]

                image = auto_canny(image)

                image = cv2.resize(image,(64,64))
                image = image[:,:,np.newaxis]
                # image = (image / 127.5) - 1.0

                images.append(image)
                angles.append(angle)
                # for i in range(3):
                #     name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                #     image = cv2.imread(name)
                #     angle = base_angle + CORRECTION[i]
                #     images.append(image)
                #     angles.append(angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = train_generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = validation_generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 80, 320  # Trimmed image format

# images = []
# measurements = []
# for line in lines:
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('/')[-1]
#         current_path = 'data/IMG/' + filename
#         image = cv2.imread(current_path)
#         images.append(image)
#         measurement = float(line[3])
#         measurements.append(measurement + correction[i])

# augmented_images = []
# augmented_measurements = []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_measurements.append(measurement * -1.0)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)


class SaveCheckpoint(Callback):
    def on_epoch_end(self, epoch, logs={}):
        #self.losses.append(logs.get('loss'))
        #y_pred = self.model.predict(self.model.validation_data[0])
        #self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        model.save('models/model-{}.h5'.format(epoch))
        return

def getModel():
    files=glob.glob("models/*.h5")
    if len(files) > 0:
        print("using previous weight files = {}".format(files))
        last_index = sorted([int(s) for s in list(filter(None,[s.replace('.h5','') for s in [s.replace('-','') for s in [s.replace('models/model','') for s in files]]]))],reverse=True)[0]
        model = load_model('models/model-{}.h5'.format(last_index))
        print("loaded model-{}.h5".format(last_index))
        return model,last_index + 1
    else:
        print("creating new model")
        model = Sequential()
        # def detect_edges(x):
        #     print("detect_edges x = {}".format(x))
        #     return cv2.Canny(x, dy=0, threshold1=100, threshold2=200)

        # model.add(Lambda(lambda x: ktf.reduce_sum(ktf.image.total_variation(x)), input_shape=(160, 320, 3)))
        # model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
        # model.add(Lambda(lambda x: K.resize_images(x, 1, 320 / 64, 'channels_first')))
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 1)))
        # model.add(Conv2D(18, (5, 5), padding='valid', activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(48, (5, 5), padding='valid', activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Dropout(0.5))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(84))
        # model.add(Dense(1))

        # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid'))
        model.add(ELU())
        model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid'))
        model.add(ELU())
        model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid'))
        model.add(ELU())

        model.add(Dropout(0.50))

        # Add two 3x3 convolution layers (output depth 64, and 64)
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(ELU())
        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(ELU())
        # model.add(Conv2D(64, (3, 3), padding='valid'))
        # model.add(ELU())
        model.add(Dropout(0.50))

        # model.add(Conv2D(64, (3, 3), padding='valid'))
        # model.add(ELU())
        # model.add(Conv2D(64, (3, 3), padding='valid'))
        # model.add(ELU())
        # model.add(Conv2D(64, (3, 3), padding='valid'))
        # model.add(ELU())
        # model.add(Dropout(0.50))


        # Add a flatten layer
        model.add(Flatten())

        # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
        model.add(Dense(100))
        model.add(ELU())
        model.add(Dropout(0.50))
        model.add(Dense(50))
        model.add(ELU())
        #model.add(Dropout(0.50))
        model.add(Dense(10))
        model.add(ELU())
        #model.add(Dropout(0.50))

        # Add a fully connected output layer
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')

        return model, 0

model, init_epoch = getModel()

# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
#xt, yt = train_generator

print("Model = {}".format(model.summary()))

history_object = model.fit_generator(train_generator, steps_per_epoch=20000 / BATCH_SIZE, validation_data=validation_generator, validation_steps=len(validation_samples) / BATCH_SIZE, initial_epoch=init_epoch, epochs=50, callbacks=[SaveCheckpoint()])

### print the keys contained in the history object
#print(history_object.history.keys())

#import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('loss_plot.png')

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

