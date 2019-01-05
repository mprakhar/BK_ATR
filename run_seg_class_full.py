# __author__ = 'Prakhar MISRA'
# Created 12/06/2017
# Last edit 12/06/2017

#Purpose:
#-----------------
#         (a) generate segmented meanshift for full PASLAR2
#         (b) capture some training segemtns and templates
#         (c) call seg_classificaiton v2 and calssfiy the segmented


import sys
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.exposure import rescale_intensity
import os
from glob import glob

# private import
import pymeanshift as pms
import seg_classificationv2
from seg_classificationv2 import class_identifyBK
import read_POLSAR as pol

pwd = os.getcwd()





# --------------------------------------------------------------------------------------------------------------------
# initalizing values for Area 1

#open the original image as array
#set files Area1
fileVV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_VV.img'
fileHV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HV.img'
fileHH = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HH.img'

imCV = np.load('baseimageA1.npy')
imCV = imCV.astype('uint8')

#training dictionary for class and corresponding segment labels -- chnage it to vector driven after words
segments_per_klass={
    5 :[53],      #water
    1 : [77, 94, 368, 54, 49, 42],     # vegetation
    2 : [438,454, 442, 431,  344, 355, 259, 264, 285, 245, 335, 154, 207 ],     #built-up urban
    3 : [60, 212, 183, 111, 76],      # BK
    4 : [2,42, 236, 260, 50, 36, 30, 83]      # surrounding BK

}

#set some templates
# AREA1 template
template1 = arrimg[97:115, 408:428] # template \
template2 = arrimg[165:215, 330:380]  # template /      #consider >.3 to
template3 = arrimg[342:380, 281:326]    # template o


# --------------------------------------------------------------------------------------------------------------------
# FULLSCENE
# initalizing values for full scene

#open the original image as array
#set files Area1
path = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Cal_Spk.data/'
fileVV = path + r'/Sigma0_VV.img'
fileHV = path + r'/Sigma0_HV.img'
fileHH = path + r'/Sigma0_HH.img'

#set texture image
texpath = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Cal_Spk_glcm.data/'
tex0 = texpath + r'/Sigma0_HV_Contrast.img'
tex1 = texpath + r'/Sigma0_HV_Dissimilarity.img'
tex2 = texpath + r'/Sigma0_HV_Homogeneity.img'
tex3 = texpath + r'/Sigma0_HV_Energy.img'



#create a baseimage by merging all bands
baseimage0 = np.zeros([3, rio.open(fileVV).read(1).shape[0], rio.open(fileVV).read(1).shape[1]])
baseimage0[0,:] = rio.open(fileHH).read(1)
baseimage0[1,:] = rio.open(fileHV).read(1)
baseimage0[2,:] = rio.open(fileVV).read(1)
bi = baseimage0

arrtex0 = np.zeros([4, rio.open(tex0).read(1).shape[0], rio.open(tex0).read(1).shape[1]])
arrtex0[0,:] = rio.open(tex0).read(1)
arrtex0[1,:] = rio.open(tex1).read(1)
arrtex0[2,:] = rio.open(tex2).read(1)
arrtex0[3,:] = rio.open(tex3).read(1)




# considering only left bottom of image as most BK there ()default is 2210, 13216
baseimage1 = baseimage0[:, 11000:16000, 0:3000]
baseimage2 = baseimage0[:, 16000:, 0:3000]
baseimage3 = baseimage0[:, 11000:16000, 3000:6500]
baseimage4 = baseimage0[:, 16000:, 3000:6500]

arrtex1 = arrtex0[:, 11000:16000, 0:3000]
arrtex2 = arrtex0[:, 16000:, 0:3000]
arrtex3 = arrtex0[:, 11000:16000, 3000:6500]
arrtex4 = arrtex0[:, 16000:, 3000:6500]



#currently using block 1
baseimage = baseimage1
arrtex = arrtex1



# set the third band as difference of other bands


baseimage[2] = (baseimage[1] - baseimage[0])/(baseimage[1] + baseimage[0])

for i in range(0, 3):
    baseimage[i, :, :] = rescale_intensity(baseimage[i, :, :], out_range=(0, 255))

#generate the Coefficient of variation and span image

baseimage = baseimage1[1]
imCV, span = pol.getCV(baseimage, win=9)
# to remove railtrack in basemage1
imCV[:, 700:825]=0
imCV[imCV>6000] = 6000
imCV = rescale_intensity(imCV, out_range=(0, 255))

#save the imCv and span
np.save('fullscene1_imCV.npy', imCV)
np.save('fullscene1_span.npy', span)

imCV = np.load('fullscene1_imCV.npy')
imCV = imCV.astype('uint8')

np.save('texraster_fullscene1.npy', arrtex1)
np.save('texraster_fullscene2.npy', arrtex2)
np.save('texraster_fullscene3.npy', arrtex3)
np.save('texraster_fullscene4.npy', arrtex4)


#setting baseimage1[1 as baseimage
#seems most likely
imCVb1 = baseimage1[1]
imCVb1[imCVb1>20] = 20
imCVb1 = rescale_intensity(imCVb1, out_range=(0, 255))
imCVb1 = imCVb1.astype('uint8')


# --------------------------------------------------------------------------------------------------------------------
# Start calling other classes
spatial_radius=6
range_radius=4.5
min_density=20
#I 6, 4.5, 90
rcmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
#(segmented_image, labels_image, number_regions) = pms.segment(imCV, spatial_radius=6,range_radius=4.5, min_density=50)
(segmented_image, labels_image, number_regions) = pms.segment(imCVb1, spatial_radius=spatial_radius,range_radius=range_radius, min_density=min_density)

np.save('segmented_meanshift_fullscene1.npy',segmented_image)
np.save('label_meanshift_fullscene1.npy',labels_image)

np.save('segmented_meanshift_fullscene1'+str(spatial_radius)+'_'+str(range_radius)+'_'+str(min_density)+'.npy',segmented_image)
np.save('label_meanshift_fullscene1_'+str(spatial_radius)+'_'+str(range_radius)+'_'+str(min_density)+'.npy',labels_image)
print('label_meanshift_fullscene1_'+str(spatial_radius)+'_'+str(range_radius)+'_'+str(min_density))



# training labels
#training dictionary for class and corresponding segment labels -- chnage it to vector driven after words
segments_per_klass={
    5 :[53],      #water
    1 : [77, 94, 368, 54, 49, 42],     # vegetation
    2 : [438,454, 442, 431,  344, 355, 259, 264, 285, 245, 335, 154, 207 ],     #built-up urban
    3 : [60, 212, 183, 111, 76],      # BK
    4 : [2,42, 236, 260, 50, 36, 30, 83]      # surrounding BK

}

# AREA1 template
#template tilted right
#creating list of sliced objects
def gettemplate():
    #template over baseimage1
    tr = [
        np.s_[1, 3599:3643, 943:1000],
        np.s_[1, 3374:3433 , 1228:1295 ],
        np.s_[1, 3354:3385 , 1557:1612 ],
        np.s_[1, 3714:3763 , 1181:1244 ],
        np.s_[1, 3671:3730 , 1460:1531 ],
        np.s_[1, 3666:3727 , 1458:1529 ],
        np.s_[1, 3068:3128 , 1185:1266 ],
        np.s_[1, 2942:2988 , 1108:1174 ]]

    tl = [
        np.s_[1, 3321:3370 , 1407:1461 ],
        np.s_[1, 3658:3700 , 1118:1172 ],
        np.s_[1, 3300:3342 , 2138:2192 ],
        np.s_[1, 3326:3372 , 1406:1461 ],
        np.s_[1, 3630:3664 , 2192:2246 ],
        np.s_[1, 2984:3024 , 481:550 ],
        np.s_[1, 4956:4997 , 2712:2776 ],
        np.s_[1, 2870:2923 , 990:1059 ],
        np.s_[1, 2793:2842 , 997:1045 ]
    ]

    to = [
        np.s_[1, 2984:3025 , 481:550 ],
        np.s_[1, 4956:4997 , 2712:2776 ]
    ]

    # save the templates
    i = 0
    for template in tr:
        templateimg = baseimage1[template]
        np.save('template_tr'+str(i)+'.npy', templateimg)
        i+=1

    i = 0
    for template in tl:
        templateimg = baseimage1[template]
        np.save('template_tl'+str(i)+'.npy', templateimg)
        i+=1

    i = 0
    for template in to:
        templateimg = baseimage1[template]
        np.save('template_to'+str(i)+'.npy', templateimg)
        i+=1


#get all templates
templatelist = []
for template in glob(os.path.join(pwd+'/template/', '*'+'.npy')):
    templatelist.append(np.load(template))


# ----------------   R U N ---------------- # ---------------------- # -------------------------

objfull = class_identifyBK(fileVV, fileHV, fileHH, segments_per_klass, arrbaseimage='fullscene1_imCV.npy',
                 arrsegmentedimage='segmented_meanshift_fullscene1.npy', arrsegmentlabel='label_meanshift_fullscene1.npy',
                 arrtexture='texraster_fullscene1.npy',  show = False)
#get the ndvi
#ndviresamp = ndvi_correct (arrndvi, arrlbl )

#set the training segements as
objfull.segments_per_klass = segments_per_klass

#set the the traing data in array adn display the segment_per_Klass
arrtrn = objfull.get_arrtrn(plot = True)

# get uniqe class names and segment labels
objfull.get_unique()

#get the max of tempate correlations
objfull.arrtemp = objfull.get_maxtemplate(templatelist)

objfull.arrtex = objfull.arrtex[1,:]
objfull.get_arrbands(totband = 4 )
objfull.get_arrbands(totband = 3 )
objfull.get_arrbands(totband = 5, allsar=True )


# get classified image
arrclf = objfull.get_classification( classifiertech = 'LR')



