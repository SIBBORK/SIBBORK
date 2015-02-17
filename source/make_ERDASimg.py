import gdal

def generate_ERDASimg_grid (metadata_file,matrix_file,numpy_raster):
    """
    this function takes the GDD matrix created in via def DefGDD, 
    combines it with metadata from the input ascii file created in GIS, 
    and outputs the GDD matrix with the geospatial metadata
    in ascii format that can be read by into GIS
    Inputs: metadata_file -- .txt file generated in GIS via conversion to ASCII tool, 
                             has georeferenced metadata
            matrix_file -- .txt file generated here by combining metadata 
                           with matrix of variables
            numpy_raster -- matrix computed in pyzelig to be displayed in GIS
    Outputs: matrix_file -- combine metadata (georef) and pyzelig computed values
    """
    format = "HFA"
    driver = gdal.GetDriverByName( format )
    src_ds = gdal.Open( metadata_file )  #the georeferencing will be taken from the metadata_file
    dst_ds = driver.CreateCopy( matrix_file, src_ds, 0 ) #the GDD matrix will be written to the matrix_file, after the georef data is copied to it
    dst_ds.GetRasterBand(1).WriteArray( numpy_raster ) #this is where whatever numpy matrix I generate in pyzelig is written to a raster to display in GIS
