# __author__ = 'Prakhar MISRA'
# Created 08/21/2017
# Last edit 09/5/2017
'''
#Purpose:
#-----------------
To provide call other classes and functions and initalize them for detection of brick kilns

Reference:


# EDIT History



#Output expected:
#-----------------
# locations of brick kilns


#Terminology used:
#-----------------
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name
'''

# import
import rasterio as rio
import numpy as np
from MSclassv2 import MulSeg


fileVV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_VV.img'
fileHV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HV.img'
fileHH = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HH.img'


file = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_3_of_VOL-ALOS2105520569-160506-HBQR1.data/merged.tif'

baseimage = np.zeros([3, rio.open(fileVV).read(1).shape[0], rio.open(fileVV).read(1).shape[1]])

baseimage[0,:] = rio.open(fileHH).read(1)
baseimage[1,:] = rio.open(fileHV).read(1)
baseimage[2,:] = rio.open(fileVV).read(1)

phase = rio.open(fileVV).read(1)

# creta an object
obj1 = MulSeg(baseimage, phase, Wc = 1, Ws = 0.5, Wt = 0.2, WD = 6, Wcs = 0.3, MSthresh = 250, set_tex = True)
#segmetn and set attributes
obj1.set_seg()
obj1.merge_seg()

texarr = obj1.texraster

#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.5, Wt = 0.4, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()


#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.9, Ws = 0.2, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()


#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.9, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()


#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.2, Wt = 0.9, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()


#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.9, Ws = 0.9, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()



#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.9, Ws = 0.2, Wt = 0.9, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()


#trying other combinations
obj2 = MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.9, Wt = 0.9, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)
obj2.texraster = obj1.texraster
obj2.set_seg()
obj2.merge_seg()






# ----------------------- multi cor processing

import glob
import os
from PIL import Image
import concurrent.futures
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from time import time

def make_image_thumbnail(filename):
    # The thumbnail will be named "<original_filename>_thumbnail.jpg"
    base_filename, file_extension = os.path.splitext(filename)
    thumbnail_filename = base_filename+'_thumbnail'+file_extension

    # Create and save thumbnail image
    image = Image.open(filename)
    image.thumbnail(size=(128, 128))
    image.save(thumbnail_filename, "JPEG")

    return thumbnail_filename

then = time()
# Loop through all jpeg files in the folder and make a thumbnail for each
for image_file in glob.glob("*.jpg"):
    thumbnail_file = make_image_thumbnail(image_file)

    print "SIMPLE A thumbnail for {image_file} was saved as {thumbnail_file} ",  time() - then

# Create a pool of processes. By default, one is created for each CPU in your machine.
then = time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Get a list of files to process
    image_files = glob.glob("*.jpg")

    # Process the list of files, but split the work across the process pool to use all CPUs!
    for image_file, thumbnail_file in zip(image_files, executor.map(make_image_thumbnail, image_files*1)):
        print "Multicore A thumbnail for {image_file} was saved as {thumbnail_file}",  time() - then

