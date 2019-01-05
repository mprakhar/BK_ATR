from PIL import Image
import numpy

OLS_subset = "E:\Acads\Research\AQM research\Data\Data_process\DMSPInterannualCalibrated_20160512\\"
DMSP_sensor = [(2001, 'F15'), (2002, 'F15'), (2003, 'F15'), (2004, 'F16'), (2005, 'F16'), (2006, 'F16'), (2007, 'F16'),
               (2008, 'F16'), (2009, 'F16'), (2010, 'F18'), (2011, 'F18'), (2012, 'F18')]
for i in DMSP_sensor:
    im = Image.open(OLS_subset+i[1]+str(i[0])+'.tif')
    # im.show()
    imarray = numpy.array(im)
    print i[1]+str(i[0]), float(sum(imarray))
