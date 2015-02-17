from osgeo import gdal
import numpy as np


def read_in_ascii(filename):
    """
	this is a function that reads in the ascii matrix from GIS
	inputs: one value for each square in the grid that corresponds to
	        the adjustment from the WMO#29481 located at 180m just 30km
			south of the forest boundary
	outputs: one value for each square in the grid that corresponds to 
	         the growing degree days for that area over the course of 
			 one year (numpy array)
    """
    dataset = gdal.Open(filename)
    raster_data = np.array(dataset.ReadAsArray(), dtype=np.float)

    return raster_data




def read_in_ascii_attributes(filename):
    """
	this is a function that reads in the ascii matrix from GIS
	inputs: filename -- the GIS .txt file containing the raster in ascii format

	outputs: raster_data -- the raster data as a numpy matrix
             x_size -- the length along the x-edge (West-East) of each pixel in the raster
             y_size -- the length along the y-edge (North-South) of each pixel in the raster
             top_left_x -- ??
             top_left_y -- ??
    """
    dataset = gdal.Open(filename)
    raster_data = np.array(dataset.ReadAsArray(), dtype=np.float)
    # (top left for X, W-E pixel resolution, ?, top left for Y, 0, N-S pixel resolution)
    NWx, x_size, blah, NWy, blah2, y_size = dataset.GetGeoTransform()  #y-size negative if heading south, so abs it later
    y_size = abs(y_size)
#	if not geotransform is None:
#	    print 'Origin = (',geotransform[0], ',',geotransform[3],')'
#		print 'Pixel Size = (',geotransform[1], ',',geotransform[5],')'
    return raster_data, x_size, y_size, NWx, NWy


