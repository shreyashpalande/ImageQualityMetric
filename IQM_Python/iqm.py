#!/usr/bin/env python
import numpy as np
import cv2
import time
from scipy.stats import entropy as scipy_entropy
import math
from scipy.signal import convolve2d
import math
from skimage.restoration import estimate_sigma

class IQM:
    def __init__(self):
        pass
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