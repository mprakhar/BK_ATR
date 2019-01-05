# __author__ = 'Prakhar MISRA'
# Created 08/22/2017
# Last edit 08/22/2017

#Purpose:
#-----------------
#         (a) To read read pre-processed POLSAR image
#         (b) To convert the image into a) CV - cofficient of variation image, span image
#                                       b) CM - covriance matrix


# EDIT History



# import
#-----------------

import numpy as np
import scipy as sci



# * * * *  * * # * * * *  * * # * * * *  * *# # * *   CV image   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#


def getCV(baseimage, win = 9):

    """function to read amplitude images to construct the CV coefficient variation image and the span image formed by summing up elemtns
    Returns CV and span image

    Parameters
    ----------
    baseimage : 3d array
        amplitude information of HH, HV, VV
    phase : float array
        phase array
    win : odd int
        window size to calculate the coefficient of caruation
    Returns
    -------
    data : float array
        CV, span
    """

    # create spane image by first taking sqaure of ampitudes to get intensity
    intns_hh = np.power(baseimage[0,:,:],2).astype(np.float32)           # stores square of amplitude of HH
    intns_hv = np.power(baseimage[1,:,:],2).astype(np.float32)           # stores square of amplitude of VV
    intns_vv = np.power(baseimage[2,:,:],2).astype(np.float32)           # stores square of amplitude of HV

    # span image
    span = np.power((intns_hh + 2 * intns_hv + intns_vv), 0.5 ).astype(np.float32)      # this is also the base image from now in this snippt

    # convert to log domain so that multiplicative noise becomes additive
    #span = (1000 * np.log(span)).astype(int) # mult by 1000 an int to decrease processing time
    span = (1000*span).astype(int)
    # remove nan
    span[span == np.nan] = 0

    # creat blank CV image. since it is based on kernel, we will now convolve to produce mean and stadard deviation
    # absed on technique suggested by https://stackoverflow.com/questions/25910050/perform-local-standard-deviation-in-python
    ones = np.ones(span.shape)
    kernel = np.ones((win, win))
    s = sci.signal.convolve2d(span, kernel, mode="same")
    s2 = sci.signal.convolve2d(span**2, kernel, mode="same")
    ns = sci.signal.convolve2d(ones, kernel, mode="same")
    # CV image after
    CV = np.sqrt((s2 - s**2 / ns) / ns)

    # remove nan values
    CV[CV==np.nan] = 0

    return CV, span

# function end


# * * * *  * * # * * * *  * * # * * * *  * *# # * *   covariance matrix   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

def getCMel(baseimage, phase):

    """function to read amplitude, phase images to covariance matrix CM for each oixel
    Returns elements of CM for each pixel

    Parameters
    ----------
    baseimage : 3d array
        amplitude information of HH, HV, VV
    phase : float array
        phase array
    win : odd int
        window size to calculate the coefficient of caruation
    Returns
    -------
    data : list of lists
        CM elements
    """

    # create spane image by first taking sqaure of ampitudes
    intns_hh = np.power(baseimage[0,:,:],2)           # stores square of amplitude of HH
    intns_vv = np.power(baseimage[1,:,:],2)           # stores square of amplitude of VV
    intns_hv = np.power(baseimage[2,:,:],2)           # stores square of amplitude of HV

    # Create ablank nd array structure
    Zc = np.zeros([5, intns_hh.shape[0],intns_hh.shape[1]], dtype=np.complex)

    #% this is gathering elements of covariance matrix, follows (Rignot and Chellappa)
    Zc[0,:,:] = intns_hh
    Zc[1, :, :] = intns_hv
    Zc[2, :, :] = intns_vv
    Zc[3,:,:] = np.array((intns_hh*intns_vv)**(.5)*np.cos(phase)) + 1j*np.array((intns_hh*intns_vv)**(.5)*np.sin(phase))
    Zc[4,:,:] = np.array((intns_hh*intns_vv)**(.5)*np.cos(-1*phase)) + 1j*np.array((intns_hh*intns_vv)**(.5)*np.sin(-1*phase))

    return Zc
# fucntione nd


def getCM(Zc):
    # this funcion will take in elements of covraianve from Zc and convert them to matrix form

    # declare 3x3 matrix
    Z = np.zeros([3,3], dtype = np.complex)
    Z[0, 0] = Zc[0]
    Z[1, 1] = Zc[1]
    Z[2, 2] = Zc[2]
    Z[0, 2] = Zc[3]
    Z[2, 0] = Zc[4]

    return Z