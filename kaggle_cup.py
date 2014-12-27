#Took from tutorial:
#https://www.kaggle.com/c/datasciencebowl/details/tutorial

#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
# make graphics inline
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))

# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
example_file = glob.glob(os.path.join(directory_names[5],"*.jpg"))[9]
# print example_file
im = imread(example_file, as_grey=True)
#plt.imshow(im, cmap=cm.gray)
#plt.show()


#Preparing the Images
#To create the features of interest, we will need to prepare the images by completing a few preprocessing procedures. We will step through some common image preprocessing actions: thresholding the images, segmenting the images, and extracting region properties. Using the region properties, we will create features based on the intrinsic properties of the classes, which we expect will allow us discriminate between them. Let's walk through the process of adding one such feature for the ratio of the width by length of the object of interest.

#First, we begin by thresholding the image on the the mean value. This will reduce some of the noise in the image. Then, we apply a three step segmentation process: first we dilate the image to connect neighboring pixels, then we calculate the labels for connected regions, and finally we apply the original threshold to the labels to label the original, undilated regions.

# First we threshold the image by only taking values greater than the mean to reduce noise in the image
# to use later as a mask
f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")

sub2 = plt.subplot(1,4,2)
plt.imshow(imthr, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
sub3 = plt.subplot(1, 4, 3)
plt.imshow(imdilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

labels = measure.label(imdilated)
labels = imthr*labels
labels = labels.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(labels)
