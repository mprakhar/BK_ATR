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

# * * * *  * * # * * * *  * * # * * * *  * *# # * *   define input files   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

#open the original image as array
#set files Area1
fileVV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_VV.img'
fileHV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HV.img'
fileHH = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HH.img'

#set emplty bsae image
baseimage = np.zeros([3, rio.open(fileVV).read(1).shape[0], rio.open(fileVV).read(1).shape[1]])

#set values in baseimage
baseimage[0,:] = rio.open(fileHH).read(1)
baseimage[1,:] = rio.open(fileHV).read(1)
baseimage[2,:] = (baseimage[0,:]-baseimage[1,:])/(baseimage[0,:]+baseimage[1,:])

arrimg = np.load('baseimageA1.npy')
plt.figure()
plt.imshow(arrimg)

#open the stored segmented image
arrseg = np.load('segmented_meanshiftA1.npy')
plt.figure()
plt.imshow(arrseg)

#open the stored labels
arrlbl = np.load('label_meanshiftA1.npy')
rcmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
plt.figure()
plt.imshow(arrlbl, cmap=rcmap)

#oen the tex raster, only bands 0, 1, 3, 4 useful in distingushng
arrtex = np.load('texrasterA1.npy')

#open the ndvi raster
arrndvi = rio.open('/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/GEE/20city classification/allndvi/BK/NewDelhi_area1.tif').read(1)


def ndvi_correct (arrndvi, arrlbl ):
    #fucntion to correct ndvi for Area1

    #rotate
    rotndvi = scipy.ndimage.interpolation.rotate(arrndvi, -10.25)
    cropndvi = rotndvi[17:82, 11:121]


    imgarrayR = skimage.transform.resize(cropndvi, np.shape(arrlbl))

    return imgarrayR


def get_arrbands(arrimg, arrndvi, arrtemp, arrtex, totband = 4, allsar = True  ):
    #  create new image with desired bands

        a = [totband]
        a.extend(list(np.shape(arrimg)))

        #create new n bnanded array with same size as arrimg
        arrbands = np.zeros(a)



        if allsar:

            #set image
            arrbands[0:3,:, :] = arrimg

            #set ndvi
            #arrbands[1, :] = arrndvi

            # set template
            arrbands[3, :, :] = arrtemp

            #set texture
            arrbands[4:totband,:,:] = arrtex

        else:
            #set image
            arrbands[0,:] = arrimg

            #set ndvi
            #arrbands[1, :] = arrndvi

            # set template
            arrbands[1, :] = arrtemp

            #set texture
            arrbands[2:totband,:] = arrtex

        return arrbands


#create a new image dataset with all the datset as badns
arrimgbands = [arrimg, arrtex] #,arrndvi]


# * * * *  * * # * * * *  * * # * * * *  * *# # * *   read the training data   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

#array with ground truth  pixels already marked with clasees
arrtrn = []

#training dictionary for class and corresponding segment labels -- chnage it to vector driven after words
segments_per_klass={
    5 :[53],      #water
    1 : [77, 94, 368, 54, 49, 42],     # vegetation
    2 : [438,454, 442, 431,  344, 355, 259, 264, 285, 245, 335, 154, 207 ],     #built-up urban
    3 : [60, 212, 183, 111, 76],      # BK
    4 : [2,42, 236, 260, 50, 36, 30, 83]      # surrounding BK

}

# set the pixels of traingn segments with corresponding classes
def get_arrtrn(dict_trn):
    #create one temporarily
    arrtrn = np.zeros(arrlbl.shape)

    #running for each class
    for i in range(1,6):
        #checking each traning lbel under the class
        for trlbl in dict_trn[i]:
            #assiging class num in the arrtrn
            arrtrn[arrlbl==trlbl] = i

    return arrtrn

def plot_trn(arrtrn):

    plt.figure()
    plt.imshow(arrtrn)
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5,6])

def get_unique(arrtrn, arrlbl):
    #function to find unique classes and segment labels
    #all classes
    num_cls = np.unique(arrtrn)
    num_cls = num_cls[1:]

    # number of labels
    segment_id = np.unique(arrlbl)

    return [num_cls, segment_id ]



# we transform each training segment into a segment model and thus creating the training dataset.
def segment_features(segment_pixels):
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

def create_segmentinfo(num_cls, arrbands, arrlbl, segment_id, segments_per_klass ):
    #create onfo for each segment; also for training segments;  by clalling function segment_features

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        objects = []
        objects_ids = []
        for segment_label in segment_id:
            segment_pixels = arrbands[:,arrlbl == segment_label]
            segment_model = segment_features(segment_pixels)
            objects.append(segment_model)
            # Keep a reference to the segment label
            objects_ids.append(segment_label)

        print("Created %i objects" % len(objects))

    # Subset the training data
    training_labels = []
    training_objects = []
    for klass in num_cls:
        class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in segments_per_klass[klass]]
        training_labels += [klass] * len(class_train_objects)
        print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
        training_objects += class_train_objects

    return [objects, objects_ids, training_objects , training_labels ]

def plot_BKrectangle(arrimg, arrlbl, objects_ids, predicted ):
    #plot the rectangle over the predicgted BK classes

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(arrimg)

    for segment_id, klass in zip(objects_ids, predicted):
        if klass == 3:
            firstpix = np.where(arrlbl==segment_id)

            rect = plt.Rectangle((firstpix[1][0], firstpix[0][0]), 10, 10, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def get_classification(arrimg, arrlbl, objects, objects_ids, training_objects, training_labels, classifiertech = 'RF'):
    #perform final classification

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
    clf = np.copy(arrlbl)

    for segment_id, klass in zip(objects_ids, predicted):
        clf[clf==segment_id] = klass

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    ax1.imshow(arrimg, interpolation='none')
    ax1.set_title('Original image')
    ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand((6), 3)))
    ax2.set_title('Classification')

    plot_BKrectangle(arrimg, arrlbl, objects_ids, predicted)

    return clf

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

def get_maxtemplate(arrimg, template_list):
    ls = []
    for template in template_list:
        temp_prob = get_templatematch(arrimg, template)
        ls.append(temp_prob)

    ls_max = np.nanmax(ls, axis=0)

    ls_max = np.where(ls_max<.3, 0, ls_max)

    return ls_max



# ----------------   R U N ---------------- # ---------------------- # -------------------------
#get the ndvi
ndviresamp = ndvi_correct (arrndvi, arrlbl )

#set the the traing data
arrtrn = get_arrtrn(segments_per_klass)

#plot the training data to confirm
plot_trn(arrtrn)

# get uniqe class names and segment labels
[num_cls, segment_id ] = get_unique(arrtrn, arrlbl)

#set some templates
# AREA1 template
template1 = arrimg[97:115, 408:428] # template \
template2 = arrimg[165:215, 330:380]  # template /      #consider >.3 to
template3 = arrimg[342:380, 281:326]    # template o
arrtemp = get_maxtemplate(arrimg, [template1, template2, template3])

# ontour and hOG
#get_contour(arrimg, 30)
#get_hog(image)

#set all bands in one array
arrtex = arrtex[1,:]
get_arrbands(arrimg, arrndvi, arrtemp, arrtex, totband = 4 )
arrbands = get_arrbands(arrimg, ndviresamp, arrtemp, arrtex, totband = 3 )
arrbands = get_arrbands(baseimage, ndviresamp, arrtemp, arrtex, totband = 5, allsar=True )

#find infor for each segment
[objects, objects_ids, training_objects , training_labels ] = create_segmentinfo(num_cls, arrbands, arrlbl, segment_id, segments_per_klass )

# get classified image
arrclf = get_classification(arrimg, arrlbl, objects, objects_ids, training_objects, training_labels, classifiertech = 'LR')
