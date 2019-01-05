# __author__ = 'Prakhar MISRA'
# Created 08/21/2017
# Last edit 08/21/2017

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



#Output expected:
#-----------------
# merged image


#Terminology used:
#-----------------
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name


# import
#-----------------

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as alg
import winsound
from scipy import ndimage as ndi
from skimage import data
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import rank
from skimage.future import graph
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries

# private
import read_POLSAR as pol

# open the camera image
image = data.camera()





# * * * *  * * # * * * *  * * # * * * *  * *# # * *   Wishart distance   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

#
def wish_dist(Wi,Wj,k):

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

    # sum of covariance matrices
    Wij = Wi+Wj

    # log of determinant probably
    log_j = np.ln(Wj(1,1)*Wj(2,2)*Wj(3,3)*(1-(np.real(Wj(1,3))^2)));   # this is actually: log(det(Wj)) >>Using analytically reduced form  (Rignot and Chellappa, 1992)
    log_i = np.ln(Wi(1,1)*Wi(2,2)*Wi(3,3)*(1-(np.real(Wi(1,3))^2)));   # this is actually: Log(Wi) >>Using analytically reduced form  (Rignot and Chellappa, 1992)
    log_ij =  np.ln(Wij(1,1)*Wij(2,2)*Wij(3,3)*(1-(np.real(Wij(1,3))^2)))

    # absolute of trace of inverse matrices
    tri = np.abs(alg.trace(alg.pinv(Wj)*Wi))
    trj = np.abs(alg.trace(alg.pinv(Wi)*Wj))

    if k ==1:
        # default Wishart distance
        dist = log_j + tri

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
# function end



class MulSeg():
    def __init__(self, baseimage, phase):

        # decare basic
        self.baseimage = baseimage
        self.phase = phase


    # * * * *  * * # * * * *  * * # * * * *  * *# # * *   Multi resolution segmentaiton   # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def merge_attr(g,i,j, image=image, labels=labels):

        # Li, Lj are parameters of of i,j
        # Ni, Nj are toal elements in i,j
        Li = g.node[i]['perimeter']
        Lj = g.node[j]['perimeter']
        Ni = g.node[i]['area']
        Nj = g.node[j]['area']

        # perimter of merged segment
        red_labels = np.copy(labels)
        red_labels[red_labels==j]=1
        red_labels[~(red_labels == i)] = 0
        Lij = find_boundaries(red_labels, mode='outer').astype(np.uint8).sum()

        # total pixel in merged segment
        Nij = Ni + Nj

        # for the texture of merged segments
        # making a fused segment by assigning iu to the label j
        labels[labels==j]=i
        # getting skimage priperties
        props = regionprops(labels, intensity_image = image)
        # finding texture over the extracted fused merged segment
        sig_tex_ij = texture(props[i].intensity_image)

    def hetero_score(g, i, j, image=image, labels=labels, Wc=1, Ws=0, Wt=0, WD=1, Wcs = .7):

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
        Returns
        -------
        data : float
            S_merge_score
        """



        # tex is the textural image containg value for eacb pixel

        # get properties of the of the nodes

        # Li, Lj are parameters of of i,j
        # Ni, Nj are toal elements in i,j
        Li = g.node[i]['perimeter']
        Lj = g.node[j]['perimeter']
        Ni = g.node[i]['area']
        Nj = g.node[j]['area']


        # need to make an object that stores PolSAR properties, textural as well
        # get covariuance matrix
        Wi  =  g.node[i]['covar']
        Wj =  g.node[j]['covar']
        #  shape measure

        # compactness measure
        #_, Lc = common_edge(labels, i, j) # only intereste d in length of common edge

        # perimter of merged segment
        red_labels = np.copy(labels)
        red_labels[red_labels==j]=1
        red_labels[~(red_labels == i)] = 0
        Lij = find_boundaries(red_labels, mode='outer').astype(np.uint8).sum()

        # total pixel in merged segment
        Nij = Ni + Nj
        # score : compute score
        Hc = Lij / (Nij) ^ .5 - (Li / (Ni) ^ .5 + Lj / (Nj) ^ .5)

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
        Bij = 2*(np.abs(np.min(Bxyi[0],Bxyj[0])-np.max(Bxyi[2],Bxyj[2])) + np.abs(np.min(Bxyi[1],Bxyj[1])-np.max(Bxyi[3],Bxyj[3])))
        # compute score
        Hs = Nij * Lij / Bij - (Ni * Li / Bi + Nj * Lj / Bj)

        # score : shPE MEASURE
        H_shape = Wcs * Hc + (1 - Wcs) * Hs

        # score : Wishart measure
        H_wish = Ni * Nj / (Ni + Nj) * (wish_dist(Wi, Wj, WD))

        # textural measure
        # find textural properties of i, j and combiedn by invoking a fuction that takes full textured image, labels and requried label
        # if multiple labels are given, then it combiens all tthe labels
        # ????? sig_tex_j = label_tex(tex, labels, [j]) create etx
        sig_tex_i = g.node[i]['tex']
        sig_tex_j = g.node[j]['tex']

        # for the texture of merged segments
        # making a fused segment by assigning iu to the label j
        labels[labels==j]=i
        # getting skimage priperties
        props = regionprops(labels, intensity_image = image)
        # finding texture over the extracted fused merged segment
        sig_tex_ij = texture(props[i].intensity_image)

        # score: texturela measure ???? please specify
        H_tex = Nij * sig_tex_ij - (Ni * sig_tex_i + Nj * sig_tex_j)

        # the lesser the merge score is, the more chancer for two segmentsto get merged
        S_merge_score = Wc*H_wish + Ws*H_shape + Wt*H_tex

        return S_merge_score
    # fucntion end

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  common edge between i and j # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    def common_edge(labels, i, j):
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
        red_labels = np.copy(labels)

        # label image reduced only to i and j
        red_labels[~((red_labels == i) | (red_labels == j))] = 0
        # get all booundaries. note that boudmary between i and 0 will lie outside i
        red_bd = find_boundaries(red_labels, mode='outer').astype(np.uint8)
        # boundary is the is those red_bd that lie within red_labels
        boundary = (red_labels * red_bd>0)

        # return boudnary as well the lenght of common boundary
        return boundary, int(boundary.sum()/2)

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  nodal covariance matrix and node texture # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def node_covar(Zc, labels,i):
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
        meanZc = np.mean(Zc[np.where(labels==i)], axis = 0)

        #arrange into matrix format
        meanZ = pol.getCM(meanZc)

        return meanZ
    # fucntion end

    def node_texture(texture, labels, i):
        """function to get the mean covrainac matrix for a label from covariancematrix of each elelemtnof a node
        Returns 3 mean texture elements - contrast, dissimlariyt, correlation

        Parameters
        ----------
        texture : float 3Ximshape array
            elements of texture
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
        meantex = np.mean(texture[np.where(labels == i)], axis=0)

        return meantex[0]. meantex[1]. meantex[2]
    # fucntion end


    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  upadte/apply attributes # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#


    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  upadte/apply attributes # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def set_attr(image, phase, labels, g):
        """function to set attributes from label properties into all nodes of g. Properties considered are
         textural measures, perimeter, area, covariance matrices
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
        props = regionprops(labels, intensity_image = image)

        # fidn the elements of covariance amtrix for all pixel coordinates
        Zc = pol.getCMel(image, phase)

        #texraster = texture(image)

        # run for each node i
        for i in g.node.keys():

            # image of bounding box elements
            node_image = props[i].intensity_image

            # raster of texture vales for that label
            texraster = texture(node_image)

            g.node[i]['perimeter'] = props[i].perimeter
            g.node[i]['area'] = props[i].area
            g.node[i]['bbox'] = props[i].bbox
            g.node[i]['covar'] = node_covar(Zc, labels,i)
            g.node[i]['contrast'], g.node[i]['dissimilarity'], g.node[i]['correlation'] = node_texture(texraster, labels, i)

        return g

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  Watershed algo # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    #http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html
    # for semi-supervised segmentation http://emmanuelle.github.io/a-tutorial-on-segmentation.html

    def im_watershed(self, image):
        # denoise image
        denoised = rank.median(image, disk(2))

        # find continuous region (low gradient -
        # where less than 10 for this image) --> markers
        # disk(5) is used here to get a more smooth image
        markers = rank.gradient(denoised, disk(5)) < 10
        markers = ndi.label(markers)[0]

        # local gradient (disk(2) is used to keep edges thin)
        gradient = rank.gradient(denoised, disk(2))

        # process the watershed
        self.labels = watershed(gradient, markers)

        # display results
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title("Original")

        ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
        ax[1].set_title("Local Gradient")

        ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
        ax[2].set_title("Markers")

        ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
        ax[3].set_title("Segmented")

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()

        return labels
    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  Texture Feature # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    # http://geoinformaticstutorial.blogspot.jp/2016/02/creating-texture-image-with-glcm-co.html

    # Gabor texture banks http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html

    def texture(image):

        PATCH_SIZE = 5 # therefore windows size = 2*Patchsize +1

        #sarraster is satellite image, testraster will receive texture
        texraster =  np.zeros([3, image.shape[0], image.shape[0]])


        for i in range(image.shape[0] ):
            print i,
            for j in range(image.shape[1] ):

                #windows needs to fit completely in image
                if i <PATCH_SIZE or j <PATCH_SIZE:
                    continue
                if i > (image.shape[0] - (PATCH_SIZE+1)) or j > (image.shape[0] - (PATCH_SIZE+1)):
                    continue

                #Calculate GLCM on a 7x7 window
                glcm_window = image[i-PATCH_SIZE: i+PATCH_SIZE+1, j-PATCH_SIZE : j+PATCH_SIZE+1]
                glcm = greycomatrix(image = glcm_window, distances = [1], angles = [0], levels= 256,  symmetric = True, normed = True )

                #Calculate contrast and replace center pixel
                # http: // scikit - image.org / docs / dev / api / skimage.feature.html  # skimage.feature.greycoprops
                texraster[0, i, j] = greycoprops(glcm, 'contrast')
                texraster[1, i, j] = greycoprops(glcm, 'dissimilarity')
                texraster[2, i, j] = greycoprops(glcm, 'correlation')

        sarplot = plt.imshow(texraster, cmap = 'gray')
        return texraster


    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  ImRAG # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#

    def set(image, phase):

        # get Cv image
        image = pol.getCV(baseimage, win=9)

        # get the labels through watershed
        labels = im_watershed(image)

        # Generate the RAG
        g = graph.rag_mean_color(image, labels)

        # set attributes of all nodes
        g = set_attr(baseimage, phase, labels, g0)

        # get neighbors
        #g.neighbors(3)

        #merge nodes i,j . optional specfy weight function
        #g.merge_nodes(4,5, weight_func=???)


        #to accecss properties of none n use g.[n]
        # to reset weight between neighbors i,j: g[i].get(j)['weight'] = K
        Freq = 2500  # Set Frequency To 2500 Hertz
        Dur = 1000  # Set Duration To 1000 == 1 second
        winsound.Beep(Freq, Dur)

        return g, labels
    # function end

    # * * * *  * * # * * * *  * * # * * * *  * *# # * *  Merge # * * * *  * * # * * * *  * * # * * * *  * *# # ** * *  * * # * * * *  * * # * * * *  * *#
    # http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html

    def mer_seg():

        g, labels = set(image, phase)

        #  finding size of each segment and then starting from the smallest
        ls = []
        for i in g.node.keys():
            ls.append([i,g.node[i]['pixel count']])

        #sorting on the pixel xount , which is the second elembt in each list
        ls = sorted(ls, key = lambda x:x[1])

        for nodel in ls:
            i = nodel[0]

            #calculating pairwise merge scores with all neigbors of i
            score_i = 0
            for j in g.neighbors(i):
                score_i.append(j, hetero_score(g, i, j, image=image, labels=labels, Wc=1, Ws=0, Wt=0, WD=1, Wcs=.7))

            # find the closest neighbor of i
            score_i = sorted(score_i, key = lambda x:x[1])
            min_i = score_i[0][0]

            # check if i is also the closest to j
            score_j = []
            for k in g.neighbors(min_j):
                score_j.append(k, hetero_score(g, min_j, k, image=image, labels=labels, Wc=1, Ws=0, Wt=0, WD=1, Wcs=.7))

            # find the closest neighbor of i
            score_j = sorted(score_j, key = lambda x:x[1])
            min_j = score_j[0][0]

            if min_i==min_j:
                src = min_j
                dst = min_i
                new = dst
                merge_attr(g,i,i) # ???

                # taking a union of the neighbors of src and dst
                src_nbrs = set(g.neighbors(src))
                dst_nbrs = set(g.neighbors(dst))
                neighbors = (src_nbrs | dst_nbrs) - set([src, dst])

                for neighbor in neighbors:

                    # finding the hetero_score between new and its neighbors
                    #data = {'ms_weight': hetero_score(g, new, neighbor, image=image, labels=labels, Wc=1, Ws=0, Wt=0, WD=1, Wcs=.7)}
                    data ={'ms_weight':-9999}
                    # setting data into in attribute dictionary
                    g.add_edge(neighbor, new, attr_dict=data)

                # the new node will now contain all properties of src and dst
                g.node[new]['labels'] = (g.node[src]['labels'] + g.node[dst]['labels'])

                # removing the individual properties of single nodes from rag
                g.remove_node(src)


                # updating the labels in the labels image
                labels[labels==src] = new






