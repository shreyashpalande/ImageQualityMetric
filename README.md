# About IQM #
This repository provides an implementation of Image Quality Metric to evaluate the quality of image for robust robot vision. 
This metric is inspired form the work done by [Shin et al.](https://arxiv.org/abs/1907.12646) along with modifications and new metrics added to it. 

The python example contains the test script to check the image quality metric on [TID2008 dataset](https://computervisiononline.com/dataset/1105138669). The extensively used Image Quality Assessment approaches available in the literature such as [BLIND](https://ieeexplore.ieee.org/document/6172573) and [BRISQUE](https://ieeexplore.ieee.org/document/6272356) gives image scores for image quality but does not provide insights about the individual attributes of the image such as noise, gradient, brightness, etc. Our approach takes image as an input and provides image quality along with noise, brightness, contrast, gradient information and entropy in the image.

This is the ongoing work which is motivated to solve the scenarios of poor features/illumination conditions for robot vision. The objective of this work is to provide robust visual localization accuracy.

# Dependencies #
1. Matplotlib 
2. Numpy 
3. cv2-library 
4. Scipy 
5. Skimage 
6. Matplotlib 

# Running with terminal #

In terminal go to the IQM_Python folder and type

```
$ python test.py
```