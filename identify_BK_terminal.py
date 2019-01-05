#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 12/02/2017
# Last edit 12/02/2017
'''
# Output expected:
#-----------------
# program to run from terminak and find locations of brick kilns


#Terminology used:
#-----------------
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name
'''



# ===============++++++++++   ------- to RUN THE CODE type -----   +++++===========
'''
python identify_BK_terminal.py  --Wc 0.2 --Ws 0.4 --Wt 0.5 --WD 6 --Wcs .2 --MSthresh 250 --set_tex False

'''

import argparse

#getting infor from the parser
parser = argparse.ArgumentParser()
parser.add_argument("--Wc", help="spectral weight")
parser.add_argument("--Ws", help="shape weight")
parser.add_argument("--Wt", help="texture weight")
parser.add_argument("--WD", help=" distance measure")
parser.add_argument("--Wcs", help=" x = smoothnes, 1-x = compactness ")
parser.add_argument("--MSthresh", help=" value fo MS threshold to be used")
parser.add_argument("--set_tex", help="generated new texture? false will use old file")

a = parser.parse_args()


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


# the real code begins here
# ----------------------------------------------------------------------------------------------------

# The real code begins now
# import
import rasterio as rio
import numpy as np
from MSclassv2 import MulSeg
import cPickle

#set files
fileVV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_VV.img'
fileHV = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HV.img'
fileHH = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_0_of_ALOS2-HBQR1_5RUA-ORBIT__ALOS2105520569-160506_Spk.data/Amplitude_HH.img'


file = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/PALSAR2/Subset/subset_3_of_VOL-ALOS2105520569-160506-HBQR1.data/merged.tif'

#set emplty bsae image
baseimage = np.zeros([3, rio.open(fileVV).read(1).shape[0], rio.open(fileVV).read(1).shape[1]])

#set values in baseimage
baseimage[0,:] = rio.open(fileHH).read(1)
baseimage[1,:] = rio.open(fileHV).read(1)
baseimage[2,:] = rio.open(fileVV).read(1)

# no phase for right now.
phase = rio.open(fileVV).read(1)

# creta an object
obj1 = MulSeg(baseimage, phase, Wc = int(a.Wc), Ws = int(a.Ws), Wt = int(a.Wt), WD = int(a.WD), Wcs = int(a.Wcs), MSthresh = int(a.MSthresh), set_tex = (a.set_tex=='True'))

#obj1 = MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.5, Wt = 0.4, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False)

objset = [
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.2, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.8, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.2, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.8, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.8, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.2, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 250, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.2, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 50, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.8, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 50, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.2, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 50, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.8, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 50, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.8, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 50, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.2, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 50, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.2, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 500, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.8, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 500, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.2, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 500, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.2, Ws = 0.8, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 500, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.8, Wt = 0.2, WD = 6, Wcs = 0.6, MSthresh = 500, set_tex = False),
MulSeg(baseimage, phase, Wc = 0.8, Ws = 0.2, Wt = 0.8, WD = 6, Wcs = 0.6, MSthresh = 500, set_tex = False)
]


i = 0
for obj1 in objset:
    #segmetn and set attributes
    start_time = time.time()
    obj1.texraster = np.load('texrasterA1.npy')
    obj1.set_seg()

    #obj1.g = cPickle.load(open('imgraph.pkl','rb'))


    #npw do the merging
    obj1.merge_seg()
    obj1.merge_seg()
    obj1.merge_seg()
    obj1.merge_seg()
    obj1.merge_seg()
    i = i+1
    print i
    print' TIME CONSIDERED - ', (time.time() - start_time)


# trial
plt.figure()
plt.imshow(tex[0])

