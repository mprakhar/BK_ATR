# __author__ = 'Prakhar MISRA'
# Created 08/21/2017
# Last edit 09/5/2017

#Purpose:
#-----------------
#         (a) To read pre-processed PolSAR iamge
#           To create functions for:
#         (b) watershed algo
#         (c) textural features
#         (c) Image Region Adjacency graph
#         (d) Hierarchical merge, Multi resolution segm merge scores
#         (e) Merging according to strategy

# EDIT History
#9/4/17 finally made it run adter correcting merging module
#12/7/17 followed mean shift instead of watershed from https://github.com/fjean/pymeanshift

#Output expected:
#-----------------
# merged image


#Terminology used:
#-----------------
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name


# import
#-----------------

import copy

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as alg
from scipy import ndimage as ndi
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import rank
from skimage.future import graph
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries
from skimage.segmentation import random_walker
#from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage
import progressbar
import time
import seaborn as sns
import math
import matplotlib
# private
import read_POLSAR as pol


# open the camera image
#image = data.camera()





# * * * *  * * # * * * *  * * # * * * *  * *# # * *   Wishart distance   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

#
def wish_dist(Zi,Zj,k, polsar=False):

    """function to find the Wishart between two covariance matrice
    Returns distance

    Parameters
    ----------
    Wi,Wj : comaplex amtrix
        covariance matrices
    k : int
        distance definition type to be used for calculation.
    Returns
    -------
    data : float
        distance
    """

    if polsar:
        # sum of covariance matrices
        Zij = Zi+Zj

        # ??????  make this in [pythin format
        # log of determinant probably
        log_j = np.log((Zj[0,0]*Zj[1,1]*Zj[2,2]*(1-(np.real(Zj[0,2])**2))))  # this is actually: log(det(Zj)) >>Using analytically reduced form  (Rignot and Chellappa, 1992)
        log_i = np.log((Zi[0,0]*Zi[1,1]*Zi[2,2]*(1-(np.real(Zi[0,2])**2))))  # this is actually: Log(Zi) >>Using analytically reduced form  (Rignot and Chellappa, 1992)
        log_ij =  np.log((Zij[0,0]*Zij[1,1]*Zij[2,2]*(1-(np.real(Zij[0,2])**2))))

        # absolute of trace of inverse matrices
        tri = np.abs(np.trace(alg.pinv(Zj)*Zi))
        trj = np.abs(np.trace(alg.pinv(Zi)*Zj))

        if k ==1:
            # default Wishart distance
            dist = np.abs(log_j + tri)
        if k == 2:
            # symmetric Wishart distance
            dist =.5*(log_i + log_j + tri + trj)
        if k == 3:
            # Bartlett distance
            dist = 2*log_ij - log_i - log_j
        if k == 4:
            # revised Wishart distance
            dist =  log_j - log_i + tri
        if k == 5:
            # another dstance
            dist =  tri + trj
        return dist

    else:
        if k== 6:
            # Bhattacharya distance: real distance sum of covariance matrices; considering mean and std dev. this option to be selcetd if node_realcovar used
            ui = Zi[0]
            sigi = Zi[1]
            uj = Zj[0]
            sigj = Zj[1]

            # mean sig
            sig = (sigi + sigj) / 2
            # default Bhattacharya distancw
            try:
                dist = 1 / 8 * ((ui - uj) * alg.inv(sig) * np.transpose(ui - uj)) + 0.5 * np.log(
                    alg.det(sig) / np.sqrt(alg.det(sigi) * alg.det(sigj)))
                if dist<0:
                    print 'negative value'
                if np.isnan(dist):
                    dist = 99999
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in err.message:
                    print 'singular matrix error'
                dist = 99999

            return np.real(dist)


# function end


# * * * *  * * # * * * *  * * # * * * *  * *# # * *  Texture Feature # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
# http://geoinformaticstutorial.blogspot.jp/2016/02/creating-texture-image-with-glcm-co.html

# Gabor texture banks http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html

def texture(image, PATCH_SIZE = 5):

    # therefore windows size = 2*Patchsize +1

    #setting up progress bar
    bar = progressbar.ProgressBar(max_value=image.shape[0])

    #sarraster is satellite image, testraster will receive texture
    texraster =  np.zeros([6, image.shape[0], image.shape[1]])


    for i in range(image.shape[0] ):
        time.sleep(0.1)
        bar.update(i)

        for j in range(image.shape[1] ):

            #windows needs to fit completely in image
            if i <PATCH_SIZE or j <PATCH_SIZE:
                continue
            if i > (image.shape[0] - (PATCH_SIZE+1)) or j > (image.shape[1] - (PATCH_SIZE+1)):
                continue

            #Calculate GLCM on a 7x7 window
            glcm_window = image[i-PATCH_SIZE: i+PATCH_SIZE+1, j-PATCH_SIZE : j+PATCH_SIZE+1]
            glcm = greycomatrix(image = glcm_window, distances = [1], angles = [0], levels= 256,  symmetric = True, normed = True )

            #Calculate contrast and replace center pixel
            # http: // scikit - image.org / docs / dev / api / skimage.feature.html  # skimage.feature.greycoprops
            texraster[0, i, j] = greycoprops(glcm, 'contrast')
            texraster[1, i, j] = greycoprops(glcm, 'dissimilarity')
            texraster[2, i, j] = greycoprops(glcm, 'correlation')
            texraster[3, i, j] = greycoprops(glcm, 'homogeneity')
            texraster[4, i, j] = greycoprops(glcm, 'energy')
            texraster[5, i, j] = greycoprops(glcm, 'ASM')

    #sarplot = plt.imshow(texraster, cmap = 'gray')
    return texraster
#function end

def node_texture(texraster, labels, i):
    """function to get the mean texture for a label from texture of each elelemtnof a node
    Returns 3 mean texture elements - contrast, dissimlariyt, correlation

    Parameters
    ----------
    texraster : float 3Ximshape array
        correation, dissimilaryt etc elements of texture for the full image
    label : int array
        contains labels of all segments
    i: int
        node number/key

    Returns
    -------
    data : 3 elements
        mean texture contrast, dissimlariyt, correlation vlaue
    """

    # mean texture elements for labels == i
    meantex = np.mean(texraster[np.where(labels == i)], axis=0)

    return meantex
# fucntion end

class MulSeg():

    """
    Class to perform multi resolution segmentation
    """
    #dictionary of textural measures
    tex_dict = {'contrast' :0,
                'dissimilarity':1,
                'correlation':2,
                'homogeneity' : 3,
                'energy' : 4,
                'ASM' : 5}

    def __init__(self, baseimage, phase, Wc = 1, Ws = 0.5, Wt = 0.2, WD = 6, Wcs = 0.3, MSthresh = 250, set_tex = False):

        # decare basic input parameters
        self.baseimage = baseimage
        self.phase = phase

        # these need to be set else default values be used
        self.Wc = Wc #spectral weight
        self.Ws =Ws #shape weight
        self.Wt = Wt #texture
        self.WD = WD # WIshart distance to be sused
        self.Wcs = Wcs  # x:smoothnes; 1-x:compactness
        self.MSthresh = MSthresh

        # to check if we need to calculate the texture image again or not
        self.set_tex = set_tex

        #these will be declared during the course of run
        self.g = None
        self.labels = None
        self.imCV = None
        self.texraster = None
        self.Zc = None



    # * * * *  * * # * * * *  * * # * * * *  * *# # * *   Multi resolution segmentaiton   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def hetero_score(self, i, j, texmes = 'contrast'):

        """function to caluclate multi resolution scheme based heterogeniety score based on hypothetical multi resolution segmentation merging
        between two neghbring segment i,j
        Returns multi-resolution score called MS score

        Parameters
        ----------
        g : graph object
            contains information about each node; eg. labels, textural info, length, area
        i,j : int
            the pair nodes for which MS to be calculated
        Wc, Ws, Wt = int
            coefficients for the merge equation
        WD : int
            wishart distance to be sued
        texmes : choose amongst 'contrast', 'dissimilarity', 'correlation'
            tex measure
        Returns
        -------
        data : float
            S_merge_score
        """

        g = copy.copy(self.g)
        image = self.imCV
        labels = copy.copy(self.labels)

        # tex is the textural image containg value for eacb pixel

        # get properties of the of the nodes

        # Li, Lj are parameters of of i,j
        # Ni, Nj are toal elements in i,j
        Li = find_boundaries(self.labels==i, mode='outer').sum()
        Lj = find_boundaries(self.labels==j, mode='outer').sum()
        Ni = g.node[i]['area']
        Nj = g.node[j]['area']
        # perimter of merged segment
        Lij = find_boundaries((self.labels==i)|(self.labels==j), mode='outer').sum()
        # total pixel in merged segment
        Nij = Ni + Nj


        # need to make an object that stores PolSAR properties, textural as well
        # get covariuance matrix
        Zi  =  g.node[i]['covar']
        Zj =  g.node[j]['covar']
        #  shape measure

        # compactness measure
        #_, Lc = common_edge(labels, i, j) # only intereste d in length of common edge


        # score : compute score
        Hc = Lij/(Nij)**0.5  - (Li * (Ni)**.5 + Lj * (Nj)**0.5)/ Nij

        # smoothness measure
        # this requires finding of border length (perimeter) of minimum enclosing rectangle X coordinates of a pixel list
        # of a label can be sorted and the difference between min and max will give the X boundary similar thing can be
        # done for the Y coordinate. Thus we have a X and Y sides of the minimum enclosing rectangle

        # Finding lenght of individual and combined bounding boxes
        # i
        # Bxyi = props[i].bbox
        Bxyi = g.node[i]['bbox']
        Bi = 2*(np.abs(Bxyi[0]-Bxyi[2]) + np.abs(Bxyi[1]-Bxyi[3]))
        # j
        # Bxyj = props[j].bbox
        Bxyj = g.node[j]['bbox']
        Bj = 2*(np.abs(Bxyj[0]-Bxyj[2]) + np.abs(Bxyj[1]-Bxyj[3]))
        # ij
        Bij = 2*(np.abs(np.min([Bxyi[0],Bxyj[0]])-np.max([Bxyi[2],Bxyj[2]])) + np.abs(np.min([Bxyi[1],Bxyj[1]])-np.max([Bxyi[3],Bxyj[3]])))
        # compute score
        Hs = Lij / Bij - (Ni * Li / Bi + Nj * Lj / Bj)/Nij

        # score : shPE MEASURE
        H_shape = self.Wcs * Hc + (1 - self.Wcs) * Hs

        # score : Wishart measure
        H_wish = Ni * Nj / (Ni + Nj) * (wish_dist(Zi, Zj, self.WD))

        # score: TEXTURAL measure
        # find textural properties of i, j and combiedn by invoking a fuction that takes full textured image, labels and requried label
        # if multiple labels are given, then it combiens all tthe labels
        # ????? sig_tex_j = label_tex(tex, labels, [j]) create etx
        sig_tex_i = g.node[i][texmes]
        sig_tex_j = g.node[j][texmes]

        # for the texture of merged segments
        # getting skimage priperties
        #props = regionprops(red_labels, intensity_image = image)
        # finding texture over the extracted fused merged segment
        #texrasterij = texture(props[i].intensity_image)
        #texrasterij = self.texraster[self.tex_dict[texmes]][np.where(self.labels == red_labels)]
        texrasterij = self.texraster[self.tex_dict[texmes]][np.where(self.labels == j)]

        # texure emasure for the merged segment
        #sig_tex_ij = node_texture(texrasterij, red_labels, i)
        sig_tex_ij = np.std(texrasterij)

        # score: texturela measure ???? please specify
        H_tex = sig_tex_ij - (Ni * sig_tex_i + Nj * sig_tex_j)/Nij

        # SCORE: the lesser the merge score is, the more chancer for two segmentsto get merged
        S_merge_score = self.Wc*H_wish + self.Ws*H_shape + self.Wt*H_tex

        return S_merge_score
    # fucntion end

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  common edge between i and j # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    def common_edge(self, i, j):
        """function to get location and lenght of common edge.
        Condition - i and j should be neighbors
        Returns list and integer.

        Parameters
        ----------
        labels : int array
            contains labels of all segments
        i,j: int
            neighboring node number/key

        Returns
        -------
        data : list, integer
            location of comon edge, length
        """

        # make copy
        red_labels = np.copy(self.labels)

        # label image reduced only to i and j
        red_labels[~((red_labels == i) | (red_labels == j))] = 0
        # get all booundaries. note that boudmary between i and 0 will lie outside i
        red_bd = find_boundaries(red_labels, mode='outer').astype(np.uint8)
        # boundary is the is those red_bd that lie within red_labels
        boundary = (red_labels * red_bd>0)

        # return boudnary as well the lenght of common boundary
        return boundary, int(boundary.sum()/2)

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  nodal covariance matrix and node texture # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def node_covar(self, i):
        """function to get the mean covrainac matrix for a label from covariancematrix of each elelemtnof a node
        Returns 3x3 array

        Parameters
        ----------
        Zc : float 5Ximshape array
            elements of covariance matrix
        label : int array
            contains labels of all segments
        i: int
            node number/key

        Returns
        -------
        data : 3x3
            mean covariance matrix
        """

        # mean covariance matrix elements for labels == i
        meanZc = np.mean(self.Zc[:,np.where(self.labels==i)[0].tolist(), np.where(self.labels==i)[1].tolist()], axis = 1)

        #arrange into matrix format
        meanZ = pol.getCM(meanZc)

        return meanZ
    # fucntion end

    def node_realcovar(self, i):
        """function to get the mean and std dev of each band in the segment
        Returns 2 3element array

        Parameters
        ----------
        Zc : float 5Ximshape array
            elements of covariance matrix
        label : int array
            contains labels of all segments
        i: int
            node number/key

        Returns
        -------
        data : 3x3
            mean covariance matrix
        """

        # mean covariance matrix elements for labels == i
        meanZc = np.mean(self.Zc[:,np.where(self.labels==i)[0].tolist(), np.where(self.labels==i)[1].tolist()], axis = 1)
        meanZc = np.matrix(meanZc[0:3])
        stdZc = np.std(self.Zc[:,np.where(self.labels==i)[0].tolist(), np.where(self.labels==i)[1].tolist()], axis = 1)
        stdZc = stdZc[0:3]
        covZc = np.zeros([3,3])
        covZc[0,0] = stdZc[0]**2
        covZc[1, 1] = stdZc[1]**2
        covZc[2, 2] = stdZc[2]**2

        return [meanZc, np.matrix(covZc)]
    # fucntion end



    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  upadte/apply attributes # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#


    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  upadte/apply attributes # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    # apply
    def set_attr(self, merge = False, dst=1):
        """function to set attributes from label properties into all nodes of g. Properties considered are
         textural measures, perimeter, area, covariance matrices. If merge = True this function will dupdate the values
         for the destiantion node dst only
        Returns updaed object g

        Parameters
        ----------
        image : float array
            base image read in the array form
        label : int array
            contains labels of all segments
        g : object
            graph.rag_mean_color(image, labels)
            

        Returns
        -------
        data : float
            S_merge_score
        """

        # find proprty of eahc label
        props = regionprops(self.labels, intensity_image = self.imCV)

        #texraster = texture(image)

        #if merge:
        #    nodelist = dst
        #else:
        #    nodelist = self.g.node.keys()

        nodelist = self.g.node.keys()
        # run for each node i
        i=-1
        for j in nodelist:
            #i = k[0]
            i = i+1
            #properties are stored in address beginning from 0 unline node/labels which start from 1
            j = nodelist[i]

            if merge:
                if j!= dst:
                    #k+=1
                    continue
            print j
            # image of bounding box elements
            #node_image = props[i].intensity_image

            # raster of texture vales for that label
            #texraster =         #texrasterij = texture(props[i].intensity_image)
            texraster = self.texraster[:, np.where(self.labels == j)[0].tolist(), np.where(self.labels == j)[1].tolist()]

            self.g.node[j]['perimeter'] = props[i].perimeter
            self.g.node[j]['area'] = props[i].area
            self.g.node[j]['bbox'] = props[i].bbox
            self.g.node[j]['covar'] = self.node_realcovar(j)
            [self.g.node[j]['contrast'], self.g.node[j]['dissimilarity'], self.g.node[j]['correlation'],
             self.g.node[j]['homogeneity'], self.g.node[j]['energy'], self.g.node[j]['ASM']] = np.std(texraster, axis = 1)
            #k+=1


    #funcot end

    # # update after a merge
    # def merge_attr(self, src, dst):
    #
    #     # updating the labels in the labels image
    #     self.labels[self.labels == src] = dst
    #
    #
    #     # getting skimage priperties
    #     props = regionprops(self.labels, intensity_image=self.imCV)
    #
    #     # finding texture over the extracted fused merged segment
    #     texraster = texture(props[dst].intensity_image)
    #
    #     # fidn the elements of covariance amtrix for all pixel coordinates
    #     #Zc = pol.getCMel(self.baseimage, self.phase)
    #
    #     self.g.node[dst]['perimeter'] = props[dst].perimeter
    #     self.g.node[dst]['area'] = props[dst].area
    #     self.g.node[dst]['bbox'] = props[dst].bbox
    #     self.g.node[dst]['covar'] = self.node_covar(dst)
    #     self.g.node[dst]['contrast'], self.g.node[dst]['dissimilarity'], self.g.node[dst]['correlation'] = self.node_texture(texraster, dst)

    def set_edge_attr(self):

        """function to set heterogeneity weight attributes between each pair of neighbors
        Returns updaed object g with added dictionary attribute called ms_weight

        Parameters
        ----------
        image : float array
            base image read in the array form
        label : int array
            contains labels of all segments
        g : object
            graph.rag_mean_color(image, labels)

        Returns
        -------
        data : float
           update ms_weight between all neighbor pairs
        """

        bar = progressbar.ProgressBar(max_value=np.shape(self.g.node.keys())[0])


        # run for each node i
        for i in self.g.node.keys():

            time.sleep(0.1)
            bar.update(i)
            # find its neihgbors
            i_nbrs = set(self.g.neighbors(i))
            # set the hetero score as 'ms_wight' in the dictioary
            for neighbor in i_nbrs:
                # finding the hetero_score between new and its neighbors
                self.g[i][neighbor]['ms_weight'] = self.hetero_score(i, neighbor)
                print self.g[i][neighbor]['ms_weight']




    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  Watershed algo # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    #http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html
    # for semi-supervised segmentation http://emmanuelle.github.io/a-tutorial-on-segmentation.html

    def im_watershed(self, Denoise_k=2, Marker_Smooth=5,  Marker_Threshold = 70, Gradient_Smooth=2):

        """function to perform watershed on the imCV - coefficient of variation image
        Returns labels for the segments

        Parameters
        ----------
        image : float array
            CV image array etc on which to perform watershed

        Returns
        -------
        label : int array
            contains labels of all segments
        """

        # denoise image
        #denoised = rank.median(self.imCV, disk(2))
        image = np.copy(self.imCV)
        image = rescale_intensity(image, out_range=(0, 1))
        denoised = rank.median(image, disk(Denoise_k))

        # find continuous region (low gradient -
        # where less than 10 for this image) --> markers
        # disk(5) is used here to get a more smooth image
        markers = rank.gradient(denoised, disk(Marker_Smooth)) < Marker_Threshold
        markers = ndi.label(markers)[0]
        #plt.imshow(markers)

        # local gradient (disk(2) is used to keep edges thin)
        gradient = rank.gradient(denoised, disk(Gradient_Smooth))

        # process the watershed
        self.labels = watershed(gradient, markers)
        #plt.figure()
        #plt.imshow(self.labels, cmap=plt.cm.spectral, interpolation='nearest')
        #plt.show()

        # display results

        # creating a random color map
        rcmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(self.imCV, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title("Original")

        ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
        ax[1].set_title("Local Gradient")

        ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
        ax[2].set_title("Markers")

        ax[3].imshow(self.imCV, cmap=plt.cm.gray, interpolation='nearest')
        ax[3].imshow(self.labels, cmap=rcmap, interpolation='nearest', alpha=.7)
        ax[3].set_title("Segmented"+str(Denoise_k)+'_'+ str(Marker_Smooth)+'_'+str(Marker_Threshold)+'_'+str(Gradient_Smooth))

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()

        plt.close()

    #fintion end



    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  ImRAG # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def set_seg(self):

        """function to set attributes and intialize all variabels of the object --> CVimage, generate watershed onto it,
            genertae its RAG, set attributesof all nodes including the neighbors
            Returns updaed object g

        Parameters
        ----------
        image : float array
            base image read in the array form


        Returns
        -------
        data : float
            S_merge_score
        """

        # get Cv image
        #self.WD =6

        self.baseimage[2] = self.baseimage[1] - self.baseimage[0]
        for i in range(0,3):
            self.baseimage[i,:,:] =  rescale_intensity(self.baseimage[i,:,:], out_range=(0, 255))

        self.imCV, span = pol.getCV(self.baseimage, win=9)
        self.imCV = rescale_intensity(self.imCV, out_range=(0, 255))

        # get the labels through watershed
        # old - self.im_watershed(Denoise_k=1, Marker_Smooth=5,  Marker_Threshold = 10, Gradient_Smooth=2)
        #new
        self.im_watershed(Denoise_k=0, Marker_Smooth=7, Marker_Threshold=9, Gradient_Smooth=2)

        #potential
        #self.im_watershed(Denoise_k=0, Marker_Smooth=5,  Marker_Threshold = 7, Gradient_Smooth=50)
        #self.im_watershed(Denoise_k=0, Marker_Smooth=6, Marker_Threshold=11, Gradient_Smooth=2)
        #self.im_watershed(Denoise_k=0, Marker_Smooth=7,  Marker_Threshold = 9, Gradient_Smooth=2)


        self.labels0 = copy.copy(self.labels)
        # tried 40

        # Generate the RAG
        self.g = graph.rag_mean_color(self.imCV, self.labels)

        #find texture for whole image if no prior texture image provieded
        if self.set_tex:

            self.texraster = texture(self.imCV.astype(np.uint8), PATCH_SIZE = 21)
            #normalize all texture
            for i in range(0,6):
                self.texraster[i,:,:] =  rescale_intensity(self.texraster[i,:,:], out_range=(0, 255))


        # fidn the elements of covariance amtrix for all pixel coordinates
        self.Zc = pol.getCMel(self.baseimage, self.phase)

        # set attributes of all nodes
        self.set_attr()

        #start_time = time.time()
        #print' TIME CONSIDERED - ', (time.time() - start_time)

        # set heterogeneity weigths scores between each neighbor
        self.set_edge_attr()




        # get neighbors
        #g.neighbors(3)

        #merge nodes i,j . optional specfy weight function
        #g.merge_nodes(4,5, weight_func=???)


        #to accecss properties of none n use g.[n]
        # to reset weight between neighbors i,j: g[i].get(j)['weight'] = K

    # function end

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  Merge # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    # http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/rag.py

    def merge_seg(self):

        """function to set attributes from label properties into all nodes of g. Properties considered are
         textural measures, perimeter, area, covariance matrices
        Returns updaed object g
        https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/graph_merge.py#L59
        https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/rag.py#L160


        Parameters
        ----------
        image : float array
            base image read in the array form
        label : int array
            contains labels of all segments
        g : object
            graph.rag_mean_color(image, labels)

        Returns
        -------
        data : float
            S_merge_score
        """

        #  finding size of each segment and then starting from the smallest
        ls = []
        for i in self.g.node.keys():
            ls.append([i,self.g.node[i]['pixel count']])

        #sorting on the pixel xount , which is the second elembt in each list
        ls = sorted(ls, key = lambda x:x[1])

        # nodel - node elements which aare being accessed from the sorted list ls
        for nodel in ls:

            # i is the label number; nodel[1] stores the pixel count
            i = nodel[0]
            #print nodel

            # since nodes are being deleted after merging, they may still apear in our original ls. hence a condition is needed
            if self.g.has_node(i):
                #calculating closest neighbor of i
                min_inbr = min([ [el, self.g[i][el]['ms_weight']] for el in self.g.neighbors(i)], key = lambda x:x[1])[0]

                # check if i is also the closest to min_inbr
                min_jnbr = min([[el, self.g[min_inbr][el]['ms_weight']] for el in self.g.neighbors(min_inbr)], key=lambda x: x[1])[0]

                if (i==min_jnbr) & (self.hetero_score(i,min_inbr)<=self.MSthresh):

                    print 'merging src ', min_inbr, ' into dst ', i, ' with merge score ', self.hetero_score(i,min_inbr)

                    src = min_inbr
                    dst = min_jnbr
                    new = dst

                    # taking a union of the neighbors of src and dst
                    src_nbrs = set(self.g.neighbors(src))
                    dst_nbrs = set(self.g.neighbors(dst))
                    neighbors = (src_nbrs | dst_nbrs) - set([src, dst])

                    # the new node will now contain all properties of src and dst
                    self.g.node[new]['labels'] = (self.g.node[src]['labels'] + self.g.node[dst]['labels'])

                    # updating the labels in the labels image
                    self.labels[self.labels == src] = dst

                    # removing the individual properties of single nodes from rag
                    self.g.remove_node(src)

                    #finding new attributes for the merged segment
                    self.set_attr( merge=True, dst=[dst])

                    for neighbor in neighbors:

                        # finding the hetero_score between new and its neighbors
                        data = {'ms_weight': self.hetero_score(new, neighbor)}
                        #data ={'ms_weight':-9999}
                        # setting data into in attribute dictionary
                        self.g.add_edge(neighbor, new, attr_dict=data)

        #plot the result
        self.display_mer()



    def display_mer(self):

        # creating a random color map
        rcmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

        # display results
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(self.baseimage[1], cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title("Original")

        ax[1].imshow(self.labels0, cmap=rcmap, interpolation='nearest')
        ax[1].set_title("Base labels")

        ax[2].imshow(self.labels, cmap=rcmap, interpolation='nearest')
        ax[2].set_title("Merged labels")

        ax[3].imshow(self.baseimage[1], cmap=plt.cm.gray, interpolation='nearest')
        ax[3].imshow(self.labels, cmap=rcmap, interpolation='nearest', alpha=.5)
        ax[3].set_title("Segmented")

        for a in ax:
            a.axis('off')

        #save the fig
        identifier = str(self.Wc)+'_'+str(self.Ws)+'_'+str(self.Wt)+'_'+str(self.WD)+'_'+str(self.Wcs)+'_'+str(self.MSthresh)
        plt.savefig('T2loc1'+identifier+'.png')

        #save the merged label image as np array
        np.save('mergedT2'+identifier+'.npy', self.labels)

        plt.close()

# pro stuff
# save texraster
# np.save('imgraph.npy', self.g)
#bb = np.load('imgraph.npy')

# pickle graph object
#filh = open('imgraph.pkl','wb')
#cPickle.dump(self.g, filh)
#bb = cPickle.load(open('imgraph.pkl','rb'))

#filh = open('baseobj.pkl','wb')
#cPickle.dump(self, filh)
#bb = cPickle.load(open('baseobj.pkl','rb'))




