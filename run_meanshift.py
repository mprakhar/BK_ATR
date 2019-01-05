# __author__ = 'Prakhar MISRA'
# Created 12/06/2017
# Last edit 12/06/2017

#Purpose:
#-----------------
#         (a) To test pymeanshift


import sys
import rasterio as rio
import numpy as np
import pymeanshift as pms
import matplotlib.pyplot as plt


imCV = np.load('baseimageA1.npy')
imCV = imCV.astype('uint8')

spatial_radius=6
range_radius=4.5
min_density=90

#(segmented_image, labels_image, number_regions) = pms.segment(imCV, spatial_radius=6,range_radius=4.5, min_density=50)
(segmented_image, labels_image, number_regions) = pms.segment(imCV, spatial_radius=spatial_radius,range_radius=range_radius, min_density=min_density)

np.save('segmented_meanshiftA1.npy',segmented_image)
np.save('label_meanshiftA1.npy',labels_image)

np.save('segmented_meanshift_'+str(spatial_radius)+'_'+str(range_radius)+'_'+str(min_density)+'.npy',segmented_image)
np.save('label_meanshift_'+str(spatial_radius)+'_'+str(range_radius)+'_'+str(min_density)+'.npy',labels_image)
print('label_meanshift_'+str(spatial_radius)+'_'+str(range_radius)+'_'+str(min_density))