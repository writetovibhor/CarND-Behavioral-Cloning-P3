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
augmentation = False

def generate_training_data(samples, output_size):
    num_samples = len(samples)
    angles = []
    if augmentation:
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

                    angles.append(angle)
                    if len(angles) == output_size:
                        return angles
    else:
        for sample in samples:
            angle = float(sample[3])
            angles.append(angle)

    return angles    

with open ('data-full/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    samples = []
    for line in reader:
        samples.append(line)

    angles = generate_training_data(samples, 20000)
    print("angles = {}".format(len(angles)))

    mean = np.mean(angles)
    sigma = np.std(angles)

    x_plot = np.linspace(min(angles), max(angles), 1000)                                                               

    fig = plt.figure()                                                               
    ax = fig.add_subplot(1,1,1)                                                      

    ax.hist(angles, bins=50, normed=True, label="data")
    ax.plot(x_plot, norm.pdf(x_plot, mean, sigma), 'r-', label="pdf")                                                          

    ax.legend(loc='best')

    x_ticks = np.arange(-4*sigma, 4.1*sigma, sigma)                                  

    ax.set_xticks(x_ticks)                                                           
    plt.savefig('histogram.png')
    plt.show() 
