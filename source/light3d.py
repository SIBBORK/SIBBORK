import numpy as np
import math
import numba

@numba.jit
def compute_3D_light_matrix(actual_leaf_area_3D_mat, 
                            dem_offset_index_mat,
                            radiation_fraction_mat,  
                            x_size, y_size, z_size,
                            list_of_grid_steps_and_proportion_tuples):
    """
    Using the Beer-Lambert law to compute the proportion of light (0 to 1) that reaches each point in
    3D space (limited to input of actual leaf area matrix dimesions).
    """
    # the size of the simulation grid
    nx, ny, nz = actual_leaf_area_3D_mat.shape

    # initialize the proportional available light matrix
    al_3D_mat = np.zeros( (nx, ny, nz) )

    # iterate through all of the "arrows" we will shoot through the "trees" to collect the
    # "number of leaves" the "arrow" passes through; these define each direction of a light
    # source as well as the proportion of light that comes from that direction
    # Note : dx, dy, dz are float values
    for dx,dy,dz,proportion in list_of_grid_steps_and_proportion_tuples:
        al_3D_mat = compute_3D_light_matrix_numba(actual_leaf_area_3D_mat, 
                                    dem_offset_index_mat,
                                    radiation_fraction_mat,  
                                    x_size, y_size, z_size,
                                    dx,dy,dz,proportion,
                                    al_3D_mat)
    return al_3D_mat

@numba.jit(nopython=True)
def compute_3D_light_matrix_numba(actual_leaf_area_3D_mat, 
                                    dem_offset_index_mat,
                                    radiation_fraction_mat,  
                                    x_size, y_size, z_size,
                                    dx,dy,dz,proportion,
                                    al_3D_mat):
    # the plot area
    plot_area = (x_size * y_size)

    # Beer-Lambert's constant coefficient for light extinction
    XK = 0.4

    # the size of the simulation grid
    nx, ny, nz = actual_leaf_area_3D_mat.shape

    # the leaf area accumulations will be scaled by the ray path length
    path_length = math.sqrt((dx*x_size)**2 + (dy*y_size)**2 + (dz*z_size)**2)
    # iterate through every x,y,z point in the grid to compute the available light from
    # the current "arrow" direction 
    for x in xrange(nx):
        for y in xrange(ny):
            for z in xrange(nz):
                # accumulate the leaf area that this "arrow" will shoot through; include wrap around
                accumulated_leaf_area = 0
                ## start accumulating 1 step away from our starting position
                #ix, iy, iz = calculate_ala_index(x, y, z,     # the current index position
                #                                 dx, dy, dz,  # the index step in each direction
                #                                 nx, ny, nz,  # the maximum index for each location
                #                                 dem_offset_index_mat) # tracks the dem offset index values under each x,y plot
                # increment the x,y positions (include wrap around)
                ax = float(x) + dx
                ay = float(y) + dy
                ix = int(round(ax) % nx)
                iy = int(round(ay) % ny)
                # For performance reasons, the leaf area columns are stored independant of the DEM.
                # This means we have to do a little math to compute where the "arrow" will lie within
                # the neighboring plot leaf area column, and then compute the z index within that column.
                az = float(z) + dz
                iz = int(round(az)) + dem_offset_index_mat[x,y] - dem_offset_index_mat[ix,iy]
                az = az + z_size * (dem_offset_index_mat[x,y] - dem_offset_index_mat[ix,iy])

                # accumulate until we hit the top of the matrix in the z direction
                while iz < nz:
                    # test if we hit ground
                    if iz < 0:
                        # The current arrow hit ground, so we're done tracing, and the actual leaf area is whatever
                        # has been accumulated thus far. Note: This approach by itself does not cause
                        # shading due to terrain (north vs south slopes), but that terrain shading will
                        # be handled in the scaling below.
                        ### Option 0: Pass through rock without ala accumulation, but continue accumulating on the other side of the rock.
                        #actual_leaf_area_val = 0.0 #comment out break to use this; 
                        #               #rock doesn't accumulate actual leaf area, but ray wraps around and continues accumulating ALA until MH (sky)

                        ## Option 1: ALA is very large when we hit a rock. This works to stop light from travelling through a rock,
                        ##            but causes valley shading when on the side of a mountain with wrap-around.
                        accumulated_leaf_area = 10e20 #comment in break; terrain breaks the ray trace, no light passes through, so total shade along this ray
                        break

                        ## Option 2 : Ray stops when we hit a rock. This could lead to bright spots right next to the rock.
                        #break #if above two lines commented out: accumulate actual leaf area along ray trace until hit terrain, but

                    # accumulate the leaf area at this location and account for the ray path length
                    actual_leaf_area_val = actual_leaf_area_3D_mat[ix,iy,iz]
                    accumulated_leaf_area = accumulated_leaf_area + (actual_leaf_area_val * path_length)
                    # move to the next location in the ray path
                    #ix, iy, iz = calculate_ala_index(ix, iy, iz,  # the current index position
                    #                                 dx, dy, dz,  # the index step in each direction
                    #                                 nx, ny, nz,  # the maximum index for each location
                    #                                 dem_offset_index_mat) # tracks the dem offset index values under each x,y plot
                    # increment the x,y positions (include wrap around)
                    ax = ax + dx
                    ay = ay + dy
                    prev_ix = ix
                    prev_iy = iy
                    ix = int(round(ax) % nx)
                    iy = int(round(ay) % ny)
                    # For performance reasons, the leaf area columns are stored independant of the DEM.
                    # This means we have to do a little math to compute where the "arrow" will lie within
                    # the neighboring plot leaf area column, and then compute the z index within that column.
                    az = az + dz
                    iz = int(round(az)) + dem_offset_index_mat[prev_ix,prev_iy] - dem_offset_index_mat[ix,iy]
                    az = az + z_size * (dem_offset_index_mat[prev_ix,prev_iy] - dem_offset_index_mat[ix,iy])

                # normalize the accumulated leaf area to per m^2 (aka leaf area index)
                lai = accumulated_leaf_area / plot_area

                # Use the Beer-Lambert law to compute the proportion of light that will reach this location.
                # Additionally, scale the available light by the terrain shading matrix pre-computed from GIS.
                al_3D_mat[x,y,z] = al_3D_mat[x,y,z] + proportion * math.exp(-XK * lai) * radiation_fraction_mat[x,y]

    return al_3D_mat


@numba.jit(nopython=True) #, inline=True)
def calculate_ala_index(x, y, z,               # the current index position
                        dx, dy, dz,            # the index step in each direction
                        nx, ny, nz,            # the maximum index for each location
                        dem_offset_index_mat): # tracks the dem offset index values under each x,y plot
    # increment the x,y positions (include wrap around)
    ix = int(round(x+dx)) % nx
    iy = int(round(y+dy)) % ny
    # For performance reasons, the leaf area columns are stored independant of the DEM.
    # This means we have to do a little math to compute where the "arrow" will lie within
    # the neighboring plot leaf area column, and then compute the z index within that column.
    reindexed_iz = int(round(z+dz)) + dem_offset_index_mat[x,y] - dem_offset_index_mat[ix,iy]

    return ix, iy, reindexed_iz



