# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:29:03 2014

@author: ZELIG
"""
import math
import numpy as np
from light3d import compute_3D_light_matrix
from read_in_ascii import read_in_ascii

def vec_scale(v, a):
    """
    Scale a 3D vector v by the scalar a.
    """
    x,y,z = v
    return np.array([a*x, a*y, a*z])

def vec_len(v):
    """
    Compute the length of a 3D vector.
    """
    x,y,z = v
    return (x**2 + y**2 + z**2)**0.5

def vec_normalize(v):
    """
    Normalize the vector to unit length.
    """
    return vec_scale(v, 1.0/vec_len(v))

def deg_to_rad(deg):
    """
    Convert degrees to radians.
    """
    return deg * math.pi / 180.

def az_deg_to_rad(azimuth):
    """
    Convert azimuth degrees to radians that can be used with cos,sin, and tan.
    This assumes that x goes west-east and y goes south-north.
    """
    rad = -1.0 * deg_to_rad(azimuth) + math.pi/2
    if rad < 0:
        rad = rad + 2*math.pi
    return rad

def elev_deg_to_rad(elevation):
    """
    Convert elevation degrees to radians that can be used with cos, sin, and tan.
    This assumes that x goes east-west and z is the vertical direction.
    """
    # Elevation angle is 0 degrees parallel to the ground and 90 degrees straight up,
    # so this follows the same convention as converting degrees to radians.
    return deg_to_rad(elevation)

def define_light_direction(azimuth, elevation):
    """
    Compute the point that will be used as the position of the light source with
    respect to the point (0,0,0). 
    The inputs azmuth & elevation represent a common angular coordinate system used
    for locating positions in the sky.
    See : http://www.esrl.noaa.gov/gmd/grad/solcalc/glossary.html#azimuthelevation
          http://www.esrl.noaa.gov/gmd/grad/solcalc/azelzen.gif

    Parameters : azimuth -- The angle in degrees measured clockwise from the north
                            to the point on the horizon directly below the object in the sky.
                            0 degrees is directly north, 90 degrees is East, 180 degrees is South
                            and 270 degrees is West.
                 elevation -- The angle in degrees measured vertically from the azimuth point
                              on the horizon up to the object in the sky. 
                              0 degrees is parallel to the ground and 90 degrees is straight up.
    
    Returns : pt -- A 3D point that represents the far end of a vector pointing from 
                    location (0,0,0) to the light source. This 'vector' will have unit length.

    Note : For this coordinate system of 3D points being (x,y,z), we assume that going in the
           positive x direction amounts to traveling south, going in the positive y direction amounts to
           traveling east, and going in the positive z direction amounts to traveling to a higher
           elevation above the ground.
    """
    # azimuth angle can go from 0 to 360 degrees
    if (azimuth < 0 or azimuth > 360):
        raise ValueError('Invalid azimuth angle (0 to 360) : received %s degrees' %(azimuth))
    # elevation angle can go from 0 to 90 degrees
    if (elevation < 0 or elevation > 90):
        raise ValueError('Invalid elevation angle (0 to 90) : received %s degrees' %(elevation))
    # from the azimuth angle compute the angle from (0,0,_) on the x-y plane
    az_rad = az_deg_to_rad(azimuth)
    #print "az_rad = ", az_rad
    # from the elevation angle compute the angle from (0,_,0) on the x-z plane
    el_rad = elev_deg_to_rad(elevation)
    #print "el_rad = ", el_rad

    r=1.0
    y = r * math.cos(az_rad)
    x = -1.0 * r * math.sin(az_rad)
    z = (x**2 + y**2)**0.5 * math.tan(el_rad)

    v = np.array([x,y,z])
    # normalize the vector to unit length
    return vec_normalize(v)
    #return v

def compute_grid_step(azimuth, elevation, xsize, ysize, zsize):
    """
    Compute the grid step that can be used to travel within a 3D grid at the direction
    specified by the input azimuth and elevation angles.
    This routine accounts for non uniform grid scales along each cardinal axes.

    The inputs azimuth & elevation represent a common angular coordinate system used
    for locating positions in the sky.
    See : http://www.esrl.noaa.gov/gmd/grad/solcalc/glossary.html#azimuthelevation
          http://www.esrl.noaa.gov/gmd/grad/solcalc/azelzen.gif

    Parameters : azimuth -- The angle in degrees measured clockwise from the north
                            to the point on the horizon directly below the object in the sky.
                            0 degrees is directly north, 90 degrees is East, 180 degrees is South
                            and 270 degrees is West.
                 elevation -- The angle in degrees measured vertically from the azimuth point
                              on the horizon up to the object in the sky. 
                              0 degrees is parallel to the ground and 90 degrees is straight up.
                 xsize,ysize,zsize -- the length along each edge of a single grid cell in meters

    Returns : grid_step_vec -- a 3 entry vector that defines how many grids (plots) to step in the x,y, and z
                               directions in order to move towards the input azimuth and elevation angles
    """
    # get a vector dx,dy,dz which is the the amount we need to travel along each
    # axis in order to point to the light source
    dir_vec = define_light_direction(azimuth, elevation) 
    # Convert the direction vector (in meters?) to a direction vector in the number of boxes (plots).
    # This conversion is necessary since we will be stepping through integer grid points and the
    # grid can have a different scale in each direction; i.e. x and y are in 10m (or 20m, or 30m, whatever the plot width & length are) steps while z is in 1m steps
    grid_dir_vec = dir_vec / np.array([xsize, ysize, zsize])
    divisor = np.min(np.abs(grid_dir_vec[np.abs(grid_dir_vec)>0]))   #want to determine the smallest possible non-zero increment (plot, m) along ray trace
    # normalize the direction vector such that the step increment in the smallest non-zero direction is 1
    dx, dy, dz = grid_dir_vec / divisor
    
    return np.array([dx,dy,dz])  #floats returned


def build_arrows_list(xsize, ysize, zsize, PHIB, PHID):
    """
    Generates a list of tuples for light ray tracing
    
    Parameters:  xsize -- x-dimension of plot (N-S axis), in m
                 ysize -- y-dimension of plot (E-W axis), in m
                 zsize -- vertical resolution, in m
                 PHIB -- proportion of on-beam direct radiation, specified in driver
                 PHID -- proportion of diffuse radiation, specified in driver

    Returns: list_of_grid_steps_and_proportion_tuples -- a list of tuples (dx, dy, dz, proportion of total radiation)
    """

    #list of tuples (azimuth angle degrees, elevation angle degrees, percent of total direct) for 57N (Usolsky)
    list_of_tuples_direct = [(0.  , 0. , 0./100.*PHIB),   (45. , 5. , 0.4/100.*PHIB),  (90. , 19., 12./100.*PHIB), 
                             (135., 38., 23.4/100.*PHIB), (180., 46., 28.4/100.*PHIB), (225., 38., 23.4/100.*PHIB),
                             (270., 19., 12./100.*PHIB),  (315., 5. , 0.4/100.*PHIB)]
    #list of tuples (azimuth angle degrees, elevation angle degrees, percent of total diffuse) for any latitude
    list_of_tuples_diffuse = [(0.  , 45., 11./100.*PHID), (90. , 45., 11./100.*PHID), (180., 45., 11./100.*PHID), 
                              (270., 45., 11./100.*PHID), (0.  , 15., 11./100.*PHID), (90. , 15., 11./100.*PHID),
                              (180., 15., 11./100.*PHID), (270., 15., 11./100.*PHID), (0.  , 90., 11./100.*PHID)]
    """
    #list of tuples (azimuth angle degrees, elevation angle degrees, NOT YET percent of total direct) for 52N (Streeline)
    list_of_tuples_direct = [(0.  , 0. , 0./100.*PHIB),   (45. , 5. , 0./100.*PHIB),  (90. , 15., 12./100.*PHIB), 
                             (135., 39., 23.4/100.*PHIB), (180., 48., 28.4/100.*PHIB), (225., 39., 23.4/100.*PHIB),
                             (270., 15., 12./100.*PHIB),  (315., 5. , 0./100.*PHIB)]
    #list of tuples (azimuth angle degrees, elevation angle degrees, NOT YET percent of total direct) for 68N (Ntreeline)
    list_of_tuples_direct = [(0.  , 0. , 0./100.*PHIB),   (45. , 2.2 , 0.4/100.*PHIB),  (90. , 11.7, 12./100.*PHIB), 
                             (135., 26.7, 23.4/100.*PHIB), (180., 32.7, 28.4/100.*PHIB), (225., 26.7, 23.4/100.*PHIB),
                             (270., 11.7, 12./100.*PHIB),  (315., 2.2 , 0.4/100.*PHIB)]
    """

    # build up a list of dx,dy,dz,proportion tuples that will be used to shoot arrows and scale the light value
    list_of_grid_steps_and_proportion_tuples = []
    for az_angle, el_angle, proportion in list_of_tuples_direct+list_of_tuples_diffuse:
        if proportion > 0 :
            dx,dy,dz = compute_grid_step(azimuth=az_angle, elevation=el_angle, xsize=xsize, ysize=ysize, zsize=zsize)
            list_of_grid_steps_and_proportion_tuples.append([dx,dy,dz,proportion])
    #print list_of_grid_steps_and_proportion_tuples
    return list_of_grid_steps_and_proportion_tuples


### INDEPENDENT PLOT MODE!!!###
def build_arrows_list_independent(xsize, ysize, zsize):
    """
    Generates a list of tuples for light ray tracing but only 1 ray from directly overhead, all light is diffuse
    
    Parameters:  xsize -- x-dimension of plot (N-S axis), in m
                 ysize -- y-dimension of plot (E-W axis), in m
                 zsize -- vertical resolution, in m

    Returns: list_of_grid_steps_and_proportion_tuples -- a list of tuples (dx, dy, dz, proportion of total radiation)
    """
    dx,dy,dz,proportion = 0.,0.,1.,1.  #directly overhead, this simplifies

    # build up a list of dx,dy,dz,proportion tuples that will be used to shoot arrows and scale the light value
    list_of_grid_steps_and_proportion_tuples = [[dx,dy,dz,proportion]]
    return list_of_grid_steps_and_proportion_tuples


def prepare_actual_leaf_area_mat_from_dem(dem_mat, zsize, max_tree_ht):
    """
    TODO: get rid of this/comment it out, b/c no need to use this anymore, since actual_leaf_area_mat is set to zeroes anyway, so removes any -1 from below ground.
    Takes the DEM text file, gets the number of grids in sim from the DEM, then pushes the actual leaf area values up 
    (ground under the trees), so calculate how many cells we need to push the actual leaf area matrix up (each x,y 
    position will be pushed up a different value, which corresponds to the elevation of terrain below this plot
    relative to the minimum elevation on the simulated grid, NOT relative to sea level).

    Parameters:  dem_mat -- matrix made from Digital Elevation Model stored in driver
                 zsize  --  z (vertical step size along tree for light computation, 1m) 
                 max_tree_ht        -- max height of tree permitted, set in driver (light subroutine doesn't check above this)

    Returns:     actual_leaf_area_mat   --  contains -1 below ground and 0 above ground for each plot and air space above plot
                                            size: nx, ny, vertical space = (max_tree_ht+(max elevation in sim - min elevation in sim))
                 dem_offset_index_mat   --  size: nx (# plots along E<-->W transect), ny (# plots along N<-->S transect), 1
                                            each x,y contains an index, which specifies the height in meters above the minimum
                                            elevation in the simulation
    """
    # get the number of cells in the z direction that will hold the plot actual leaf area data (w/o DEM adjustment)
    zcells = math.ceil(max_tree_ht / zsize)

    # get the number of grid points from the dem; the actual leaf area grid x,y dimension will be the same size
    nx, ny = dem_mat.shape
    # what is the minimum height of the dem terrain within simulated grid of plots
    dem_min = np.min(dem_mat)
    # The DEM will push the actual leaf area values up (ground under the trees), so calculate how
    # many cells we need to push the actual leaf area matrix up (each x,y position will be pushed up 
    # a different value, which corresponds to the elevation of terrain below this plot
    # relative to the minimum elevation on the simulated grid, NOT relative to sea level).
    # dtype stored as integer, so as to use these values as indices later on.
    dem_offset_index_mat = np.array( np.round((dem_mat - dem_min) / zsize), dtype=np.int) #meters to box step conversion
    additional_zcells = np.max(dem_offset_index_mat) #highest elevation on grid
    #print 'additonal z offset due to DEM is %s cells' %(additional_zcells)
    # calculate the total number of cells we need to hold the relativized DEM elevation
    # and trees that can reach up to the max_tree_ht
    total_zcells = zcells + additional_zcells
    # inititally the actual leaf area matrix is all zeros (no trees and no ground)
    actual_leaf_area_mat = np.zeros( (nx, ny, total_zcells) )  #size: nx, ny, max_tree_ht+rangeOfDEMelevs
    #print 'actual leaf area matrix shape is :' ,actual_leaf_area_mat.shape
    # fill in the below ground levels with the special flag (-1)
    GROUND_FLAG = -1
    for x in xrange(nx):
        for y in xrange(ny):
            gnd_level = dem_offset_index_mat[x,y]
            actual_leaf_area_mat[x,y,:gnd_level] = GROUND_FLAG  #everything below ground =-1
    # We should now have an actual leaf area matrix with below ground levels flagged,
    # and above ground values empty and ready to be filled. To use this,
    # we need the light routine to be aware of the ground flag, as well
    # as provide the ground offset level where each tree level begins.
    return actual_leaf_area_mat, dem_offset_index_mat


if __name__ == '__main__':  #need this to run from commandline
    from volume_slicer import VolumeSlicer

    # define the plot size along the x,y, and z dimentions
    # 10m by 10m by 1m
    xsize, ysize, zsize = 10., 10., 1.
    # the maximum height a tree can reach in m
    MAX_TREE_HT = 50
    # pre-compute the "arrows" that will be used to compute the light at every position in the actual leaf area matrix
    arrows_list = build_arrows_list(xsize=xsize, ysize=ysize, zsize=zsize, PHIB=0.45, PHID=0.55)
    # get the empty actual leaf area matrix that has been adjusted for the DEM
    actual_leaf_area_mat, dem_offset_index_mat = prepare_actual_leaf_area_mat_from_dem(dem_ascii_filename='elevation10m.txt', 
                                                             xsize=xsize,ysize=ysize,zsize=zsize, 
                                                             max_tree_ht=MAX_TREE_HT)
    #actual_leaf_area_mat[:,:,:] = 0.0
    # Read in the radiation fraction matrix that is used to scale the proportion of radiation
    # that is available to every location on the DEM at ground-level. This will be used to account for terrain shading.
    # Same fraction used for top-of-canopy, since found no significant difference b/w the two.
    radiation_fraction_mat = np.array(read_in_ascii(filename='radiationfraction_ascii.txt'), dtype=np.float)
    rad_min = np.min(radiation_fraction_mat)
    rad_max = np.max(radiation_fraction_mat) # the range is often very small, unless transect N-S?
    print 'min rad = %s max rad = %s' %(rad_min, rad_max)

    ## TEST
    # put in a tree popsicle; to remove tree comment out the two lines below
    px, py = 10, 10
    actual_leaf_area_mat[px,py, dem_offset_index_mat[px,py]+30:-10] = 1e20  #that's one dense tree!
    #px, py = 10, 11
    #actual_leaf_area_mat[px,py, dem_offset_index_mat[px,py]+30:-10] = 100000.0
    #px, py = 11, 10
    #actual_leaf_area_mat[px,py, dem_offset_index_mat[px,py]+30:-10] = 100000.0
    #px, py = 11, 11
    #actual_leaf_area_mat[px,py, dem_offset_index_mat[px,py]+30:-10] = 100000.0
    ## END TEST

    #setup the actual_leaf_area_mat shape based on new nz, which includes terrain and max tree height "head space"
    nx,ny,nz = actual_leaf_area_mat.shape
    # finally compute the available light matrix based on the actual leaf area and the "arrow" directions and proportions
    # actual_leaf_area_mat will have 0s and -1s in it during 1st year of sim, then populated with other values (-1s stay same)
    al_mat = compute_3D_light_matrix(actual_leaf_area_mat, #with -1 flags at and below ground surface
                                     radiation_fraction_mat, #scaled by max radiation received on s-facing slopes >10deg, computed in GIS
                                     nx,ny,nz, #shape of actual leaf area matrix
                                     arrows_list, #list of tuples for direct & diffuse angles for light computation
                                     plot_area=xsize*ysize) #defines the size of one plot in m^2
    print dem_offset_index_mat[0,0]
    print actual_leaf_area_mat[0,0,:]
    print al_mat[0,0,:]

    m = VolumeSlicer(data=al_mat) #doesn't work in Windows OS; shadow to the south due to diffuse light component
    m.configure_traits()

    #example()


