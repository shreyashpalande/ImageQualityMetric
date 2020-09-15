#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from pathlib import Path
import warnings
import iqm as image_quality

warnings.filterwarnings("ignore")
path = Path().absolute()

w1 = w2 = w3 = w4 = 0.5
w5 = 0.2

image_set1 = [str(path)+"/I01_16_4.bmp",
              str(path)+"/I01_16_2.bmp",
              str(path)+"/I01.BMP",
              str(path)+"/I01_16_1.bmp",
              str(path)+"/I01_16_3.bmp"]

fig = plt.figure(1)
fig.suptitle('Door IQM Metric for (Changing Brightness)', fontsize=20)

row, col, count = 2, 5, 1
for im in image_set1:
    t0 = time.time()
    iqm = image_quality.IQM()
    image_raw = cv2.imread(im)
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_RGB2GRAY)

    brightness, contrast = iqm.image_brightness(image_gray, None)

    entropy = iqm.image_entropy(image_gray)
    Lgradient, gradient_image = iqm.gradient_mapping(I=image_gray)

    awgn = iqm.estimate_noise(image_gray)
    _, NoiseImage, noise = iqm.estimate(image_gray)
    Noise = (noise*0.5+awgn*0.5)/20

    iqm_metric = w1*brightness+w2*contrast+w3*entropy+w4*Lgradient-w5*Noise

    fig.add_subplot(row, col, count), plt.imshow(
        cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)), plt.title("Image")
    plt.text(0, 500, r'IQM Metric=%.2f' %
             (iqm_metric), fontsize=20, color='red')
    fig.add_subplot(row, col, count+5)
    plt.imshow(gradient_image, cmap='gray')
    plt.title("Gradient Image")

    count = count+1

image_set1 = [str(path)+"/I01_17_4.bmp",
              str(path)+"/I01_17_2.bmp",
              str(path)+"/I01.BMP",
              str(path)+"/I01_17_1.bmp",
              str(path)+"/I01_17_3.bmp"]
fig = plt.figure(2)
fig.suptitle('Door IQM Metric for (Changing Contrast)', fontsize=20)

row, col, count = 2, 5, 1
for im in image_set1:
    t0 = time.time()
    iqm = image_quality.IQM()
    image_raw = cv2.imread(im)
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_RGB2GRAY)

    brightness, contrast = iqm.image_brightness(image_gray, None)

    entropy = iqm.image_entropy(image_gray)
    Lgradient, gradient_image = iqm.gradient_mapping(I=image_gray)

    awgn = iqm.estimate_noise(image_gray)
    _, NoiseImage, noise = iqm.estimate(image_gray)
    Noise = (noise*0.3+awgn*0.7)/10

    iqm_metric = w1*brightness+w2*contrast+w3*entropy+w4*Lgradient-w5*Noise

    fig.add_subplot(row, col, count), plt.imshow(
        cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)), plt.title("Image")
    plt.text(0, 500, r'IQM Metric=%.2f' %
             (iqm_metric), fontsize=20, color='red')
    fig.add_subplot(row, col, count+5)
    plt.imshow(gradient_image, cmap='gray')
    plt.title("Gradient Image")

    count = count+1

image_set1 = [str(path)+"/I01.BMP",
              str(path)+"/I01_08_1.bmp",
              str(path)+"/I01_08_2.bmp",
              str(path)+"/I01_08_3.bmp",
              str(path)+"/I01_08_4.bmp"]
fig = plt.figure(3)
fig.suptitle('Door IQM Metric for (Changing Gradient)', fontsize=20)

row, col, count = 2, 5, 1
for im in image_set1:
    t0 = time.time()
    iqm = image_quality.IQM()
    image_raw = cv2.imread(im)
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_RGB2GRAY)

    brightness, contrast = iqm.image_brightness(image_gray, None)

    entropy = iqm.image_entropy(image_gray)
    Lgradient, gradient_image = iqm.gradient_mapping(I=image_gray)

    awgn = iqm.estimate_noise(image_gray)
    _, NoiseImage, noise = iqm.estimate(image_gray)
    Noise = (noise*0.3+awgn*0.7)/10

    iqm_metric = 0.5*brightness+0.5*contrast+0.2*entropy+2.*Lgradient-0.2*Noise

    fig.add_subplot(row, col, count), plt.imshow(
        cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)), plt.title("Image")
    plt.text(0, 500, r'IQM Metric=%.2f' %
             (iqm_metric), fontsize=20, color='red')
    fig.add_subplot(row, col, count+5)
    plt.imshow(gradient_image, cmap='gray')
    plt.title("Gradient Image")

    count = count+1

image_set1 = [str(path)+"/I01.BMP",
              str(path)+"/I01_01_1.bmp",
              str(path)+"/I01_01_2.bmp",
              str(path)+"/I01_01_3.bmp",
              str(path)+"/I01_01_4.bmp", ]
fig = plt.figure(4)
fig.suptitle('Door IQM Metric for (Changing Noise)', fontsize=20)

row, col, count = 2, 5, 1
for im in image_set1:
    t0 = time.time()
    iqm = image_quality.IQM()
    image_raw = cv2.imread(im)
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_RGB2GRAY)

    brightness, contrast = iqm.image_brightness(image_gray, None)

    entropy = iqm.image_entropy(image_gray)
    Lgradient, gradient_image = iqm.gradient_mapping(I=image_gray)

    awgn = iqm.estimate_noise(image_gray)
    _, NoiseImage, noise = iqm.estimate(image_gray)
    Noise = (noise*0.3+awgn*0.7)/10

    iqm_metric = w1*brightness+w2*contrast+w3*entropy+w4*Lgradient-w5*Noise

    fig.add_subplot(row, col, count), plt.imshow(
        cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)), plt.title("Image")
    plt.text(0, 500, r'IQM Metric=%.2f' %
             (iqm_metric), fontsize=20, color='red')
    fig.add_subplot(row, col, count+5)
    plt.imshow(gradient_image, cmap='gray')
    plt.title("Gradient Image")

    count = count+1
plt.show()
