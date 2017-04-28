import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
from math import exp, fabs, ceil
from scipy.stats import norm
import sklearn


angles = []
CORRECTION = [0.0, 0.25, -0.25]
MAX_OFFSET = 25.0
batch_size = 64
augmentation = True

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

def generate_training_data(samples, output_size):
    num_samples = len(samples)
    angles = []
    plt_count = 1
    if augmentation:
        fig = plt.figure(figsize=(16, 10))
        for iters in range(ceil(output_size / batch_size)):
            samples = sklearn.utils.shuffle(samples)
            batch_samples = samples[0:batch_size]

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
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image)
                    plt_count += 1
                    #cv2.imwrite('temp/{}-0.png'.format(angle), image)

                    image = image[70:135, 0:320]
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image)
                    plt_count += 1
                    # cv2.imwrite('temp/{}-1-crop.png'.format(angle), image)

                    image = augment_brightness_camera_images(image)
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image)
                    plt_count += 1
                    # cv2.imwrite('temp/{}-2-brightness.png'.format(angle), image)

                    image = add_random_shadow(image)
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image)
                    plt_count += 1
                    # cv2.imwrite('temp/{}-3-shadow.png'.format(angle), image)

                    if do_flip:
                        image = cv2.flip(image,1)
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image)
                    plt_count += 1
                    # cv2.imwrite('temp/{}-4-flip.png'.format(angle), image)

                    image = auto_canny(image)
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image, cmap=plt.get_cmap('gray'))
                    plt_count += 1
                    # cv2.imwrite('temp/{}-5-canny.png'.format(angle), image)

                    M = np.float32([[1,0,x_offset],[0,1,y_offset]])
                    rows = image.shape[0]
                    cols = image.shape[1]
                    image = cv2.warpAffine(image,M,(cols,rows))
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image, cmap=plt.get_cmap('gray'))
                    plt_count += 1
                    # cv2.imwrite('temp/{}-6-translate.png'.format(angle), image)

                    image = cv2.resize(image,(64,64))
                    sub = plt.subplot(10, 8, plt_count)
                    sub.set_xticks(())
                    sub.set_yticks(())
                    plt.imshow(image, cmap=plt.get_cmap('gray'))
                    plt_count += 1
                    # cv2.imwrite('temp/{}-7-resize.png'.format(angle), image)
                    image = image[:,:,np.newaxis]


                    angles.append(angle)
                    if len(angles) == output_size:
                        fig.tight_layout()
                        plt.savefig('preprocessing.png')
                        return angles
        fig.tight_layout()
        plt.savefig('preprocessing.png')
    else:
        for sample in samples:
            angle = float(sample[3])
            angles.append(angle)

    return angles    

with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    samples = []
    for line in reader:
        samples.append(line)

    angles = generate_training_data(samples, 10)
    print("angles = {}".format(len(angles)))
