#!/usr/bin/env python
# -*- coding: utf-8 -*-
#__author__ = 'Prakhar MISRA'
# Created 2/12/2016
#Last edit 8/12/2017

'''# Purpose: This code will read the ENVI files that have aready been processed at Monthly level (by specifying the kind of file MODIS and the type of data product ) in batch mode. also it will give the values for select pixel coodridinates for the time series data.
# the purpose os not only to read the Monthly level provcessed files but also to find outliers in the data acquisition
# Location of output: E:\Acads\Research\AQM\Data process\Code results

#Contents:
# 1. zonal_stats(input_zone_polygon, input_value_raster) :　 to return the stats of a shapefile area in raster
# 2. loop_zonal_stats(input_zone_polygon, input_value_raster): to run the zonal_stats functions for varoius shapefiles
# 3. zone_mask(input_zone_polygon, input_value_raster): to return just the mask
# 4. arr_to_raster(numpy_arr): to convert the numpy array data back to raster

History:
8/12/17 Added rasterio functionality
'''

from osgeo import gdal, gdalnumeric, ogr, osr
#from PIL import Image, ImageDraw
import os, sys
import numpy as np
import pandas as pd
import infoFinder as info
import fiona as fio
import rasterio as rio
import rasterio.mask
import coord_translate as ct
from rasterio import Affine
from rasterio.warp import reproject, Resampling






# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
# this code is a modification of https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#calculate-zonal-statistics

# type gdalinfo .tif to find info on a file also the coordinate system
def zonal_stats(input_zone_polygon, input_value_raster):

    # Open data
    driver = gdal.GetDriverByName('ENVI')   #envi
    driver.Register()                       #envi


    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1], options= ["ALL_TOUCHED=TRUE"])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

    #diplay mask
    # img=Image.fromarray(datamask, 'L')
    # img.show()
    # print datamask, 'sum is ', numpy.sum(datamask), numpy.min(datamask),numpy.max(datamask), numpy.count_nonzero(datamask), numpy.size(datamask)
    # if you divide the total light by size(total area) you get numpy.average and if you divide it by nonzero count area you get mean which is what we will eventually use

    # Mask zone of raster
    zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
    zoneraster[zoneraster<0]=0
    # raising to power 2.0/3.0 to convert DN to radiance
    #zoneraster = numpy.power(zoneraster,2/3)
    # Calculate statistics of zonal raster
    return ((np.nansum(zoneraster),np.nanmean(zoneraster),np.nanmedian(zoneraster),np.nanstd(zoneraster)))




# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
def loop_zonal_stats(input_zone_polygon, input_value_raster, ID):

    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    statDict = {}
    lyr_ID=0

    for FID in featList:
        feat = lyr.GetFeature(FID)
        meanValue = zonal_stats(input_zone_polygon, input_value_raster)
    for feature in lyr:
        lyr_ID = feature.GetField(ID)


    #print meanValue
    toret=(lyr_ID, meanValue)
    print (toret)
    return toret
# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
def all_stt(input_value_raster, shapefile_path,ID, varsfx):  # full india scorecard :)
    # input_value_raster
    # shapefile_path - dir of all the shape file
    # df_shpatt - List of all shape file. making dataframe of shape file attribute list
    # out_file - output name
    # ID -  shapefile feature which needs to statisticed
    # varsfx - any suffix after new variable added

    ind2 = []
    dl_sum=[]
    dl_mean=[]
    dl_med=[]
    dl_std=[]
    sd=os.path.normpath(input_value_raster)

    #opening each vector and storing their info
    for vector in info.findVectors(shapefile_path, '*.shp'):
        (vinfilepath, vinfilename)= os.path.split (vector)
        input_zone_polygon = vinfilepath+ '/' + vinfilename
        sd=os.path.normpath(vector)
        id, (sum, mean, med, std)  = list(loop_zonal_stats(input_zone_polygon, input_value_raster, ID))

        ind2.append(id)
        dl_sum.append(sum)
        dl_mean.append(mean)
        dl_med.append(med)
        dl_std.append(std)


    d = {ID:ind2, 'sum'+varsfx:dl_sum,'mean'+varsfx:dl_mean}#,'med'+varsfx:dl_med,'std'+varsfx:dl_std }
    df = pd.DataFrame(d)

    return df



# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
# given a shapefile and a raster, this will return the mask that is to be applied over the raster
def zone_mask(input_zone_polygon, input_value_raster, bound = 0):

    # Open data
    driver = gdal.GetDriverByName('ENVI')   #envi
    #driver = gdal.GetDriverByName('GTiff')
    driver.Register()                       #envi


    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1], options= ["ALL_TOUCHED=TRUE"])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

    #diplay mask
    # img=Image.fromarray(datamask, 'L')
    # img.show()
    # Mask zone of raster
    zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
    #creating amasked array, http://docs.scipy.org/doc/numpy/reference/routines.ma.html; the original data is still present but behind the mask

    return zoneraster, datamask







# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
#Source: http://blog.remotesensing.io/2013/03/using-gdal-with-python-basic-intro/#more
# to convert array back to raster
def arr_to_raster(dataarray, input_value_raster, out_raster ):
    # dataarray is the dataarray to be convert to raster
    # input_value_raster is the is the file whose geospatial attributes will be copied into dataraster
    # output_raster is the location of the output raster file

    # Read the input raster into a Numpy array
    data = gdal.Open(input_value_raster)

    # First of all, gather some information from the original file
    [cols, rows] = dataarray.shape
    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    # nodatav = data.GetNoDataValue()
    outfile = out_raster

    # Create the file, using the information from the original file
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(str(outfile), rows, cols, 1, gdal.GDT_Float32)

    # Write the array to the file, which is the original array in this example
    outdata.GetRasterBand(1).WriteArray(dataarray)

    # Set a no data value if required
    # outdata.GetRasterBand(1).SetNoDataValue(nodatav)

    # Georeference the image
    outdata.SetGeoTransform(trans)

    # Write projection information
    outdata.SetProjection(proj)



# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
#Source: http://blog.remotesensing.io/2013/03/using-gdal-with-python-basic-intro/#more
# to convert array back to raster
def ndarr_to_raster(dataarray, input_value_raster, out_raster ):
    # dataarray is the dataarray to be convert to raster
    # input_value_raster is the is the file whose geospatial attributes will be copied into dataraster
    # output_raster is the location of the output raster file

    # Read the input raster into a Numpy array
    data = gdal.Open(input_value_raster)

    # First of all, gather some information from the original file
    [cols, rows,nd] = dataarray.shape
    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    # nodatav = data.GetNoDataValue()
    outfile = out_raster

    # Create the file, using the information from the original file
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(str(outfile), rows, cols, nd, gdal.GDT_Float32)

    # Write the array to the file, which is the original array in this example
    for i in range(0,nd):
        outdata.GetRasterBand(i+1).WriteArray(dataarray[:,:,i])
    # outdata.GetRasterBand(2).WriteArray(dataarray[:, :, 1])
    # outdata.GetRasterBand(3).WriteArray(dataarray[:, :, 2])

    # Set a no data value if required
    # outdata.GetRasterBand(1).SetNoDataValue(nodatav)

    # Georeference the image
    outdata.SetGeoTransform(trans)

    # Write projection information
    outdata.SetProjection(proj)






# input_value_raster= r'E:\Acads\Research\AQM research\Codes\trialbox\trial_viirs.tif'
# # # raster_path= r'E:\Acads\Research\AQM research\Codes\trialbox'
# # # raster_path= r'D:/75N060E'
# # # shapefile_path = 'E:\Acads\Research\AQM research\Data\Data process\Ind_adm3_splitshp'  #IND_adm3_ID_3__299.shp
# input_zone_polygon = 'E:\Acads\Research\AQM research\Data\Data process\Ind_adm3_splitshp\IND_adm3_ID_3__303.shp'
# # a =loop_zonal_stats(input_zone_polygon, input_value_raster)

def raster_as_array(rasterfilename):
    # Open data
    driver = gdal.GetDriverByName('ENVI')  # envi
    driver.Register()
    raster = gdal.Open(rasterfilename)
    banddataraster = []
    for i in xrange(1, raster.RasterCount + 1): #
        banddataraster.append(raster.GetRasterBand(i).ReadAsArray().astype(np.float))

    return banddataraster


# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *   Subset raster using vector* # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
#Source - https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/cookbook.html#masking-raster-with-a-polygon-feature

def rio_zone_mask(input_zone_polygon_path, input_value_raster_path, output_value_raster_path):
    # input_zone_polygon_path - path of plogyon or shapefile containg all polygons
    # input_value_raster_path - raster to be masked
    # feature_code - feature code of the polygon path that needs to be used for subsetting

    with fio.open(input_zone_polygon_path, "r") as shapefile:
        geoms = [feature['geometry'] for feature in shapefile]

    with rio.open(input_value_raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, geoms, all_touched=True, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rio.open(output_value_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)





# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *   Subset raster using raster* # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
# function to clip the DSM boundaried data from night light
def clipraster(raster_base, raster_shape, shape=0):

    # raster_base: path whoch needs to clipped; its path address
    # raster_shape: path of the size/shape which will be extracted out; its path address
    # shape:shape to be clipped ; get np.shape(raster_shape); has been put in case the cell size of raster_shape and shape is different

    # finding size of raster shape
    # b =  gdal.Open(raster_shape)
    # c = b.GetRasterBand(1)
    # shape = np.shape(c.ReadAsArray().astype(np.float))
    # shape = np.shape(unDSMresmnarr)

    # nDSM array
    shapearr = raster_as_array(raster_shape)[0]

    # shape of nDSMpath
    shape = np.shape(shapearr)

    #finding lat long of pixel of raster_shape
    [[lat1,long1]] = ct.pixelToLatLon(raster_shape, [[0,0]])
    [[lat2, long2]] = ct.pixelToLatLon(raster_shape, [[shape[1]-1, shape[0]-1]])

    # fidning what pixel of the raster base does the latlong obtained from raster shape correspond
    [pix1] = ct.latLonToPixel(raster_base,[[lat1,long1]] )
    [pix2] = ct.latLonToPixel(raster_base, [[lat2, long2]])

    # carve out the raster_shape from base starting coordinates 'pix'
    clippedarr = raster_as_array(raster_base)[0][pix1[1]-1: pix2[1], pix1[0]-1: pix2[0]]

    # still need to apply the mask(

    return clippedarr
# fucntion end


# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
#Source: https://mapbox.s3.amazonaws.com/playground/perrygeo/rasterio-docs/writing.html
# to convert array back to raster
def rio_arr_to_raster(dataarray, input_value_raster_path, out_raster):

    src = rio.open(input_value_raster_path)
    # context manager.
    with rio.drivers():
        # Write the product as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the meta attributes of
        # the source file, but then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile = src.profile
        profile.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')

        with rio.open(out_raster, 'w', **profile) as dst:
            dst.write(dataarray.astype(rio.float32), 1)


# * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *  # * * * *  * * # * * * *  * * # * * *
#Source: https://mapbox.github.io/rasterio/topics/reproject.html
# to resample array
def rio_resample(srcpath,  outpath, res_factor = 15):


    with rio.open(srcpath) as src:

        arr = src.read()
        newarr = np.empty(shape=(arr.shape[0], int(arr.shape[1] * res_factor), int(arr.shape[2] * res_factor)))

        # adjust the new affine transform to the 150% smaller cell size
        aff = src.affine
        newaff = Affine(aff.a / res_factor, aff.b, aff.c, aff.d, aff.e /res_factor, aff.f)

        kwargs = src.meta.copy()
        #kwargs['transform'] = newaff
        #kwargs['affine'] = newaff
        kwargs.update({
            'crs': src.crs,
            'transform': newaff,
            'affine': newaff,
            'width': int(arr.shape[2] * res_factor),
            'height': int(arr.shape[1] * res_factor)
        })


        with rio.open(outpath, 'w', **kwargs) as dst:

            reproject(
                arr, newarr,
                src_transform = aff,
                dst_transform = newaff,
                src_crs = src.crs,
                dst_crs = src.crs,
                resample = Resampling.bilinear)

            dst.write(newarr[0].astype(rio.float32), indexes=1)

