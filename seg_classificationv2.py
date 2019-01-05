# __author__ = 'Prakhar MISRA'
# Created 12/7/2017
# Last edit 12/7/2017

#Purpose:
#-----------------
#         (a) To read read the segmented image
#           To create functions for:
#         (b) appropritaely assign training data
#         (c) train the classifier
#         (c) Following https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb
#         (d)
#         (e)

# EDIT History
#


#Output expected:
#-----------------
# classified image and sts


#Terminology used:
#-----------------
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name




# import
#-----------------

import copy

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import scipy
from matplotlib import colors
from skimage.measure import block_reduce
import numpy.linalg as alg
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from skimage import measure
from skimage import transform
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import rank
from skimage.future import graph
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage
from sklearn import svm
import progressbar
import time
import seaborn as sns
import math
import warnings
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import match_template
import cv2
from scipy import ndimage as ndi
import matplotlib
from skimage import feature
from sklearn.linear_model import LogisticRegression

# private
import read_POLSAR as pol


def getdeg(arrimg):
    edges1 = feature.canny(arrimg)
    edges2 = feature.canny(arrimg, sigma=1)

def get_contour(r, dist):
    # gen contour of the edge image
    # and for perimeter of contour http://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.polygon_perimeter

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(r, dist)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return contours


#find oerimeter of each contour and only keep contours withing certin range of permeter



# Histogram of oriented grahics
#https://gurus.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
def get_hog(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()


# template matching
def get_templatematch(image, template, threshold = 0.8):
    coin = template

    result = match_template(image, coin, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2, adjustable='box-forced')

    ax1.imshow(coin, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    plt.show()

    return result

class class_identifyBK():

    def __init__(self, VVpath, HVpath, HHpath, segments_per_klass, arrbaseimage, arrsegmentedimage , arrsegmentlabel , arrtexture , show = False):

        # * * * *  * * # * * * *  * * # * * * *  * *# # * *   define input files   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
        #open the original image as array
        self.fileVV = VVpath
        self.fileHV = HVpath
        self.fileHH = HHpath

        #  set emplty bsae image
        #self.baseimage = np.zeros([3, rio.open(self.fileVV).read(1).shape[0], rio.open(self.fileVV).read(1).shape[1]])

        #  set values in baseimage
        #self.baseimage[0,:] = rio.open(self.fileHH).read(1)
        #self.baseimage[1,:] = rio.open(self.fileHV).read(1)
        #self.baseimage[2,:] = (self.baseimage[0,:]-self.baseimage[1,:])/(self.baseimage[0,:]+self.baseimage[1,:])

        self.arrimg = np.load(arrbaseimage)

        #open the stored segmented image
        self.arrseg = np.load(arrsegmentedimage)

        #open the stored labels
        self.arrlbl = np.load(arrsegmentlabel)

        #oen the tex raster, only bands 0, 1, 3, 4 [contrat, dissimilarty, homegenity, energy] useful in distingushng
        self.arrtex = np.load(arrtexture)

        #open the ndvi raster
        self.arrndvi = rio.open('/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/GEE/20city classification/allndvi/BK/NewDelhi_area1.tif').read(1)

        # create a new image dataset with all the datset as badns
        self.arrimgbands = [self.arrimg, self.arrtex]  # ,arrndvi]

        #set the training segments
        self.segments_per_klass = segments_per_klass

        #whether to show all loaded layers
        if show:
            plt.figure()
            plt.imshow(self.arrimg)

            plt.figure()
            plt.imshow(self.arrseg)

            rcmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
            plt.figure()
            plt.imshow(self.arrlbl, cmap=rcmap)


    def ndvi_correct (self):
        #fucntion to correct ndvi for Area1

        #rotate
        rotndvi = scipy.ndimage.interpolation.rotate(self.arrndvi, -10.25)
        cropndvi = rotndvi[17:82, 11:121]

        self.imgarrayR = skimage.transform.resize(cropndvi, np.shape(self.arrlbl))

        return self.imgarrayR


    def get_arrbands(self, totband = 4, allsar = True  ):
        #  create new image with desired bands

        # allsar command will consider all SAR bands else only band 0/baseimage will be considered

            a = [totband]
            a.extend(list(np.shape(self.arrimg)))

            #create new n bnanded array with same size as arrimg
            arrbands = np.zeros(a)

            if allsar:

                #set image
                arrbands[0:3,:, :] = self.arrimg

                #set ndvi
                #arrbands[1, :] = self.arrndvi

                # set template correlaion result, preferable max of several templates
                arrbands[3, :, :] = self.arrtemp

                #set texture
                arrbands[4:totband,:,:] = self.arrtex

            else:
                #set image
                arrbands[0,:] = self.arrimg

                #set ndvi
                #arrbands[1, :] = arrndvi

                # set template correlaion result, preferable max of several templates
                arrbands[1, :] = self.arrtemp

                #set texture
                arrbands[2:totband,:] = self.arrtex

            self.arrbands = arrbands




    # set the pixels of traingn segments with corresponding classes
    def get_arrtrn(self, plot = False):
        #create one temporarily
        self.arrtrn = np.zeros(self.arrlbl.shape)

        #running for each class
        for i in range(1,6):
            #checking each traning lbel under the class
            for trlbl in self.segments_per_klass[i]:
                #assiging class num in the arrtrn
                self.arrtrn[self.arrlbl==trlbl] = i

        if plot:
            self.plot_trn()


    def plot_trn(self):

        plt.figure()
        plt.imshow(self.arrtrn)
        plt.colorbar(ticks=[0, 1, 2, 3, 4, 5,6])

    def get_unique(self):
        #function to find unique classes and segment labels
        #all classes
        num_cls = np.unique(self.arrtrn)
        self.num_cls = num_cls[1:]

        # number of labels
        self.segment_id = np.unique(self.arrlbl)


    # we transform each training segment into a segment model and thus creating the training dataset.
    def segment_features(self, segment_pixels):
        """For each band, compute: min, max, mean, variance, skewness, kurtosis"""
        features = []
        n_bands, n_pixels = segment_pixels.shape
        for b in range(n_bands):
            stats = scipy.stats.describe(segment_pixels[b,:])
            band_stats = list(stats.minmax) + list(stats)[2:]
            if n_pixels == 1:
                # scipy.stats.describe raises a Warning and sets variance to nan
                band_stats[3] = 0.0  # Replace nan with something (zero)
                band_stats = band_stats[1:3] # considering only max and mean
            features+=band_stats
        return features

    #compute the features' vector for each segment (and append the segment ID as reference)
    # This is the most heavy part of the process. It could take about half an hour to finish in a not-so-fast CPU

    def create_segmentinfo(self):
        #create onfo for each segment; also for training segments;  by clalling function segment_features

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            objects = []
            objects_ids = []
            for segment_label in self.segment_id:
                segment_pixels = self.arrbands[:,self.arrlbl == segment_label]
                segment_model = self.segment_features(segment_pixels)
                objects.append(segment_model)
                # Keep a reference to the segment label
                objects_ids.append(segment_label)

            print("Created %i objects" % len(objects))

        # Subset the training data
        training_labels = []
        training_objects = []
        for klass in self.num_cls:
            class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in self.segments_per_klass[klass]]
            training_labels += [klass] * len(class_train_objects)
            print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
            training_objects += class_train_objects

        return [objects, objects_ids, training_objects , training_labels ]


    def plot_BKrectangle(self, objects_ids, predicted ):
        #plot the rectangle over the predicgted BK classes

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(self.arrimg)

        for segment_id, klass in zip(objects_ids, predicted):
            if klass == 3:
                firstpix = np.where(self.arrlbl==segment_id)

                rect = plt.Rectangle((firstpix[1][0], firstpix[0][0]), 10, 10, edgecolor='w', facecolor='none')
                ax.add_patch(rect)

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()



    def get_classification(self, classifiertech = 'RF'):
        #perform final classification

        objects, objects_ids, training_objects, training_labels = self.create_segmentinfo()

        #Train a classifier

        if classifiertech =='RF':
            classifier = RandomForestClassifier(n_jobs=-1)
            classifier.fit(training_objects, training_labels)

        if classifiertech == 'SVM'  :
            classifier = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=100.)
            classifier.fit(training_objects, training_labels)

        if classifiertech =='LR':
            # define logit object
            classifier = LogisticRegression(solver='sag', max_iter=500, random_state=42, multi_class='multinomial')
            classifier.fit(training_objects, training_labels)

        # classify everyone
        predicted = classifier.predict(objects)

        # Propagate the classification from segment to pixel
        clf = np.copy(self.arrlbl)

        for segment_id, klass in zip(objects_ids, predicted):
            clf[clf==segment_id] = klass

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        ax1.imshow(self.arrimg, interpolation='none')
        ax1.set_title('Original image')
        ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand((6), 3)))
        ax2.set_title('Classification')

        self.plot_BKrectangle(objects_ids, predicted)

        return clf



    def get_maxtemplate(self,  template_list):
        ls = []
        for template in template_list:
            temp_prob = get_templatematch(self.arrimg, template)
            ls.append(temp_prob)

        ls_max = np.nanmax(ls, axis=0)

        ls_max = np.where(ls_max<.3, 0, ls_max)

        return ls_max

