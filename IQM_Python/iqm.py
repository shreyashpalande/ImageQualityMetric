#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from scipy.stats import entropy as scipy_entropy
import math
from scipy.signal import convolve2d
import math
from skimage.restoration import estimate_sigma
from pathlib import Path
import warnings


warnings.filterwarnings("ignore")
path = Path().absolute()

w1 = w2 = w3 = w4 = 0.5
w5 = 0.2


class IQM:
    def image_entropy(self, image, base=2):
        # image = self.image_gradient(image)
        _, counts = np.unique(image, return_counts=True)
        return scipy_entropy(counts, base=base)/8.0

    def image_gradient(self, image, depth=cv2.CV_8U, size=3):
        sobelx = cv2.Sobel(image, depth, 1, 0, ksize=size,
                           scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        sobely = cv2.Sobel(image, depth, 0, 1, ksize=size,
                           scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        return cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    def gradient_mapping(self, I, lam=100, threshold=0.05):
        I = self.image_gradient(I)
        I = np.array(I, dtype=float)
        N = np.log10(lam*(1-threshold)+1)
        I = I/255.0
        I[I < threshold] = 0
        gradient = lam*(I-threshold) + 1
        gradient = np.ma.filled(np.log10(np.ma.masked_equal(gradient, 0)), 0)/N
        tiles, _, _ = self.image_slice(gradient, 1000)
        Lgradient = cv2.mean(np.array(tiles))[0]
        return Lgradient, gradient

    def image_slice(self, im, number_tiles):
        im_w = im.shape[0]
        im_h = im.shape[1]
        columns = 0
        rows = 0
        columns = int(math.ceil(math.sqrt(number_tiles)))
        rows = int(math.ceil(number_tiles / float(columns)))
        extras = (columns * rows) - number_tiles
        tile_w, tile_h = int(math.floor(im_w / columns)
                             ), int(math.floor(im_h / rows))
        tiles = []
        for pos_y in range(0, im_h - rows, tile_h):
            for pos_x in range(0, im_w - columns, tile_w):
                image = im[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w]
                tiles.append((np.sum(image)/(tile_h*tile_w)))

        return tiles, rows, columns

    def image_brightness(self, image, mask=None):
        mean, std = cv2.meanStdDev(image, mask=mask)
        return mean[0][0]/255.0, std[0][0]/255.0

    def estimate_noise(self, I):

        return estimate_sigma(I, multichannel=False, average_sigmas=False)

    def estimate(self, I, threshold=0.01):

        p = 0.10
        H, W = I.shape
        gradimage = self.image_gradient(I, cv2.CV_64FC1, 3)
        gradimage = gradimage.astype(np.uint8)

        Grad1D = np.reshape(
            gradimage, (1, gradimage.shape[0]*gradimage.shape[1]))
        sortGrad1D = np.sort(Grad1D)
        threshold = sortGrad1D[0, int(p*gradimage.shape[0]*gradimage.shape[1])]
        HomogenousRegionMask = gradimage <= threshold
        UnsaturatedMask = I.copy()
        UnsaturatedMask[(UnsaturatedMask <= 15) | (UnsaturatedMask >= 235)] = 0
        UnsaturatedMask[UnsaturatedMask != 0] = 1
        UnSaturatedHomogenousMask = UnsaturatedMask * HomogenousRegionMask

        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]

        Laplacian_Image = convolve2d(
            I.astype(np.float64), np.array(M, dtype=np.float64), 'same')

        Masked_Laplacian_Image = Laplacian_Image*UnSaturatedHomogenousMask
        Ns = np.sum(np.sum(UnSaturatedHomogenousMask))
        noise = math.sqrt(math.pi/2) * (1/(6*Ns)) * \
            np.sum(np.sum(np.absolute(Masked_Laplacian_Image)))
        if Ns < H*W*0.0001:
            noise = math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2)) * \
                np.sum(np.sum(np.absolute(Masked_Laplacian_Image)))

        return UnSaturatedHomogenousMask, Masked_Laplacian_Image.astype(np.uint8), noise


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
    iqm = IQM()
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
    iqm = IQM()
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
    iqm = IQM()
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
    iqm = IQM()
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
