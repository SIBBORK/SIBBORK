import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, round, sqrt

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def compute_3D_light_matrix(np.ndarray[DTYPE_t, ndim=3] actual_leaf_area_3D_mat not None, 
                            np.ndarray[DTYPE_t, ndim=2] radiation_fraction_mat not None, 
                            unsigned int nx, unsigned int ny, unsigned int nz, 
                            unsigned int x_size, unsigned int y_size, unsigned int z_size,
                            list_of_grid_steps_and_proportion_tuples):
    """
    Using the Beer-Lambert law to compute the proportion of light (0 to 1) that reaches each point in
    3D space (limited to input of actual leaf area matrix dimesions).
    """
    #for cython make sure to specify variable type for optimization
    cdef float GROUND_FLAG
    cdef float XK
    #cdef unsigned int nx, ny, nz
    cdef float dx, dy, dz, proportion
    cdef int x, y, z
    cdef float accumulated_leaf_area
    cdef float lai
    cdef float ax, ay, az
    cdef float actual_leaf_area_val
    cdef float path_length
    cdef float plot_area = <float> (x_size * y_size)

    # Beer-Lambert's constant coefficient for light extinction
    XK = 0.4
    # a special flag is used in the actual leaf area matrix to indicate a gound level
    GROUND_FLAG = -1.0

    # the size of the simulation grid
    #nx, ny, nz = actual_leaf_area_3D_mat.shape
    # initialize the proportional available light matrix
    cdef np.ndarray[DTYPE_t, ndim=3] al_3D_mat = np.zeros( (nx, ny, nz), dtype=DTYPE) #, order='C' )

    # iterate through all of the "arrows" we will shoot through the "trees" to collect the
    # "number of leaves" the "arrow" passes through; these define each direction of a light
    # source as well as the proportion of light that comes from that direction
    for dx,dy,dz,proportion in list_of_grid_steps_and_proportion_tuples:
        # the leaf area accumulations will be scaled by the ray path length
        path_length = sqrt((dx*x_size)**2 + (dy*y_size)**2 + (dz*z_size)**2)
        # iterate through every x,y,z point in the grid to compute the available light from
        # the current "arrow" direction 
        for x in xrange(nx):
            for y in xrange(ny):
                for z in xrange(nz):
                    # accumulate the leaf area that this "arrow" will shoot through; include wrap around
                    #accumulated_leaf_area = accumulate_leaf_area(actual_leaf_area_3D_mat, nx,ny,nz,
                    #                                             x,y,z, dx,dy,dz)
                    accumulated_leaf_area = 0
                    # start accumulating 1 step away from our starting position
                    ax = <float> x + dx
                    ay = <float> y + dy
                    az = <float> z + dz
                    ix = <unsigned int> (round(ax) % nx)
                    iy = <unsigned int> (round(ay) % ny)
                    iz = <unsigned int> round(az)

                    actual_leaf_area_val = actual_leaf_area_3D_mat[x,y,z]
                    if actual_leaf_area_val == GROUND_FLAG:
                        # us a very large value for ground actual leaf area that all light is extinct
                        accumulated_leaf_area = 10e20
                    else:
                        # accumulate until we hit the top of the matrix in the z direction
                        while iz < nz:
                            actual_leaf_area_val = actual_leaf_area_3D_mat[ix,iy,iz]
                            if actual_leaf_area_val == GROUND_FLAG:
                                # The current arrow hit ground, so we're done tracing, and the actual leaf area is whatever
                                # has been accumulated thus far. Note: This approach by itself does not cause
                                # shading due to terrain (north vs south slopes), but that terrain shading will
                                # be handled in the scaling below.
                                ### Option 1: Pass through rock without ala accumulation, but continue accumulating on the other side of the rock.
                                #actual_leaf_area_val = 0.0 #comment out break to use this; 
                                #               #rock doesn't accumulate actual leaf area, but ray wraps around and continues accumulating ALA until MH (sky)

                                ## Option 1: ALA is very large when we hit a rock. This works to stop light from travelling through a rock,
                                ##            but causes valley shading when on the side of a mountain with wrap-around.
                                #accumulated_leaf_area = 10e20 #comment in break; terrain breaks the ray trace, no light passes through, so total shade along this ray
                                #break

                                ## Option 2 : Ray stops when we hit a rock. This could lead to bright spots right next to the rock.
                                break #if above two lines commented out: accumulate actual leaf area along ray trace until hit terrain, but

                            accumulated_leaf_area = accumulated_leaf_area + (actual_leaf_area_val * path_length)
                            ax = ax + dx
                            ay = ay + dy
                            az = az + dz
                            ix = <unsigned int> (round(ax) % nx)
                            iy = <unsigned int> (round(ay) % ny)
                            iz = <unsigned int> round(az)

                    # normalize the accumulated leaf area to per m^2 (aka leaf area index)
                    lai = accumulated_leaf_area / plot_area
#                    if x==0 and y==0 and z==0:  #select one plot per year
                        #if dx==0. and dy==0. and dz==1.:  #select the overhead arrow(direction) 
#                        print "%0.1f %0.1f %0.1f LAI= %s" %(dx,dy,dz,lai)
                    # Use the Beer-Lambert law to compute the proportion of light that will reach this location.
                    # Additionally, scale the available light by the terrain shading matrix pre-computed from GIS.
                    al_3D_mat[x,y,z] = al_3D_mat[x,y,z] + proportion * exp(-XK * lai) * radiation_fraction_mat[x,y]

    return al_3D_mat

