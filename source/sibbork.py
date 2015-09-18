"""
Import all the necessary libraries and tools for the model.
"""
import numpy as np
import random
from math import pi
from weather import GrowingDegreeDays_calc, rain_sim, drydays, one_time_radiation_readin
from load_driver import load_driver_py
from LightSubroutine import build_arrows_list, prepare_actual_leaf_area_mat_from_dem, build_arrows_list_independent
from read_in_ascii import read_in_ascii, read_in_ascii_attributes
from light3d import compute_3D_light_matrix
import pandas as pd
import h5py
import dill
# Select either the realu numba (could be buggy)
import numba
# or select the python only version (used to compare to real numba version)
#import numba_dummy as numba




#TODO: change constants to all CAPS
#TODO: need to compute stress flags (right now no trees killed by SMORT)

def StartSimulation(driver_filename, output_filename, driver):
    """
    1) Puts drivers (from GIS) into a form python understands
    2) Iterates over each year

    Parameters:  driver_filename -- the name and path of the driver file 
                 output_filename -- the name and path of the output hdf file

    Returns:     None  
    """

    # set the random number seed at startup, so our random sequences should be the same every time
    if driver['CLIMATE_RANDOM'] == False:
        random.seed(0)
        np.random.seed(0)

    # initialize the matrices, state variables and constants that will be passed down from year to year
    driver, initial_soil_water_mat, \
    arrows_list, actual_leaf_area_mat, dem_offset_index_mat, radiation_fraction_mat, plot_area, \
    DBH_matrix, crown_base_matrix, species_code_matrix, \
                stress_flag_matrix, seed_bank_matrix, hdf_file = initialization(driver_filename, output_filename, driver)
    # get the dimension of each plot from the driver (this is specified in the DEM file; the path to the DEM file is specified in the driver
    # the z_size_m is the resolution with which we track the vertical dimension of the trees (at this point, 1m)
    x_size_m = int( driver['EW_plot_length_m'] )  #plot width in meters
    y_size_m = int( driver['NS_plot_length_m'] )  #plot length in meters
    z_size_m = int( driver['vertical_step_m'] )   #above ground, through the air

    #initialize the groups for HDF file (output)
    DBH_group = hdf_file["DBH"]
    SpeciesCode_group = hdf_file["SpeciesCode"]
    BasalArea_group = hdf_file["BasalArea"]
    Biomass_group = hdf_file["Biomass"]
    DBH_distribution_group = hdf_file["DBH_distribution"]
    ## Debug groups
    if driver['DEBUG']:
        # weather
        DegreeDays_group = hdf_file["DegreeDays"]                              # size: nx,ny,nspp
        RelativeDryDays_group = hdf_file["RelativeDryDays"]                    # size: nx,ny,nspp
        # intermediate results
        RelativeSoilFertility_group = hdf_file["RelativeSoilFertility"]        # size: nx,ny
        # computed factors
        GrowingDegreeDaysFactor_group = hdf_file["GrowingDegreeDaysFactor"]    # size: nx,ny,nspp
        SoilMoistureFactor_group = hdf_file["SoilMoistureFactor"]              # size: nx,ny,nspp
        SoilFertilityFactor_group = hdf_file["SoilFertilityFactor"]            # size: nx,ny,nspp
        AvailableLightFactor_group = hdf_file["AvailableLightFactor"]          # size: nx,ny,ntrees
        GrowthFactor_group = hdf_file["GrowthFactor"]                          # size: nx,ny,ntrees
        SproutFactor_group = hdf_file["SproutFactor"]                          # size: nx,ny,nspp

    #get the temporal, spatial, and species  dimensions for the model run
    sim_start_year = driver['sim_start_year']
    sim_stop_year = driver['sim_stop_year']
    spp_in_sim = number_of_species
    nx,ny,ntrees = DBH_matrix.shape

    simulation_years_logged = []

    ##specify whether to generate random weather or read it in from a file (usually an HDF output from a previous run)
    #if driver["CLIMATE_RANDOM"] == False:
    #    climate_filepath = driver['Climate_record_file_path']
    #    hdf_climate_file = h5py.File(climate_filepath, "r")
    #    Temperature_record_group = hdf_climate_file["Temperature"]
    #    Precipitation_record_group = hdf_climate_file["Precipitation"]

    for year in range(sim_start_year, sim_stop_year+1):  #+1 b/c python upper bound is not inclusive
        print "Simulation year %s" % year

        ## checking to see if need to generate random weather or read it in from a record
        #if driver["CLIMATE_RANDOM"] == False:
        #    record_year = "%.4d" % year
        #    month_simtemp_vec = Temperature_record_group[record_year][:]  #pulling out recorded temps from hdf file (previous sim)
        #    monthly_sim_rain_vec = Precipitation_record_group[record_year][:]  #pulling out recorded precip from hdf file (previous sim)
        #else:
        #    month_simtemp_vec = None  #this way generate random weather if CLIMATE_RANDOM=true in driver
        #    monthly_sim_rain_vec = None #this way generate random weather if CLIMATE_RANDOM=true in driver

        # generate weather:
        monthly_temp_mat_lst, \
        initial_soil_water_mat, \
        GDD_matrix, drydays_fraction_mat = generate_weather(driver, initial_soil_water_mat, year)
        #if year==2100:  #this and the line below are printouts for testing
        #    print "GDD mat=", GDD_matrix[:,0]
        # kill trees (stressed trees from previous years get to fear for their lives and some may die)
        DBH_matrix, stress_flag_matrix, \
        species_code_matrix, crown_base_matrix = kill_trees(DBH_matrix, species_code_matrix, stress_flag_matrix, crown_base_matrix, number_of_species)

        # compute species specific values for each tree, most of these are f(DBH)
        tree_height_matrix, \
        total_leaf_area_matrix, \
        biomass_matrix, \
        optimal_growth_increment_matrix, \
        optimal_biomass_matrix, \
        basal_area_matrix, \
        biovolume_matrix, \
        optimal_biovolume_matrix, \
        optimal_biovolume_increment_matrix = compute_individual_tree_values(DBH_matrix, species_code_matrix, crown_base_matrix)

        # compute the actual leaf area (TODO: check that this value doesn't get out of control, unrealistic, implement QC, although LAIs in the lower 20s have been reported)
        actual_leaf_area_mat = compute_actual_leaf_area(DBH_matrix, species_code_matrix, crown_base_matrix, 
                                                        tree_height_matrix, total_leaf_area_matrix, z_size_m, driver['plot_area_m2'])


        # compute the weather related factors at the plot level
        GDD_3D_spp_factor_matrix, \
        soil_moist_3D_spp_factor_matrix = compute_species_factors_weather(GDD_matrix, drydays_fraction_mat, number_of_species)


        # compute 3D light using a more efficient algorithm
        available_light_mat = compute_light(actual_leaf_area_mat, dem_offset_index_mat, 
                                            radiation_fraction_mat, arrows_list, 
                                            x_size_m,y_size_m,z_size_m)

        # compute crown base:
        crown_base_matrix = compute_crown_base(DBH_matrix, species_code_matrix, tree_height_matrix,
                                               crown_base_matrix, available_light_mat)


        # compute the ground-level and vertical profile of light factors for each species on each plot
        ground_light_3D_spp_factor_matrix, \
        available_light_spp_factor_matrix = light_factor_compute(available_light_mat,
                                                                 tree_height_matrix, crown_base_matrix,
                                                                 species_code_matrix, number_of_species)


        # based on the limits of degree day and light factors, compute what the biovolume will be on each plot
        likely_biovolume_increment_matrix = compute_plot_likely_biovolume_increment(species_code_matrix, GDD_3D_spp_factor_matrix, 
                                                                                    available_light_spp_factor_matrix, 
                                                                                    optimal_biovolume_increment_matrix)
        # compute the relative soil fertility (based on ratio of what the plot can support to likely biovolume increment)
        relative_soil_fertility_matrix = biovolume_compute_relative_soil_fertility_matrix(likely_biovolume_increment_matrix, #units: m3/tree
                                                                                          driver['biovolume_soil_fert_mat'])  #units: m3/plot


        # compute the soil/nutrition/site-index factor and the permafrost factor (size of each matrix: nx,ny,nspp)
        soil_fert_3D_spp_factor_matrix, permafrost_factor_matrix = compute_species_factors_soil(relative_soil_fertility_matrix, number_of_species, driver['permafrost_mat'])


        #when run on the local host, this is what gets printed to terminal during model run for each year
        print "**** Factor Analysis (Plot 0,0) ****"
        print "  Growing Degree Days = ", GDD_matrix[0,0]
        print "    Growing Degree Day Factors by Species: ", GDD_3D_spp_factor_matrix[0,0]
        print "  Dry Days = ", drydays_fraction_mat[0,0]
        print "    Soil Moisture Factors by Species:      ", soil_moist_3D_spp_factor_matrix[0,0]
        print "  Relative Soil Fertility = ", relative_soil_fertility_matrix[0,0]
        print "    Soil Fertility Factors by Species:     ", soil_fert_3D_spp_factor_matrix[0,0]
        max_tree_height = int(np.max( tree_height_matrix[0,0] ))
        dem_offset = dem_offset_index_mat[0,0]
        available_light_plot0 = available_light_mat[0,0,dem_offset:dem_offset+max_tree_height]
        print "  Average Available Light = ", np.mean( available_light_plot0[available_light_plot0 >= 0] )
        print "  dem offset = ", dem_offset
        print "  Max Tree Height = ", max_tree_height
        print "  Ground Light Factors by Species: ", ground_light_3D_spp_factor_matrix[0,0]


        # computing the regeneration factor for each species on each plot
        # first, compute min(smf,sff,permafrost)*gddf   (below ground factors & thermal effects)
        partial_growth_factor_matrix = np.minimum(np.minimum(soil_moist_3D_spp_factor_matrix, soil_fert_3D_spp_factor_matrix), permafrost_factor_matrix) * GDD_3D_spp_factor_matrix

        # grow trees
        DBH_matrix, stress_flag_matrix, \
        growth_factor_matrix = grow_trees(DBH_matrix, species_code_matrix, available_light_spp_factor_matrix, 
                                          partial_growth_factor_matrix, stress_flag_matrix, optimal_growth_increment_matrix,species_code_to_stress_threshold)

        # sizes of matrices: inputs: (nx,ny,spp_in_sim) * min((nx,ny,spp_in_sim),(nx,ny,spp_in_sim)) * (nx,ny,spp_in_sim)
        # sprout_factor_matrix = ground_light_3D_spp_factor_matrix * np.minimum(soil_moist_3D_spp_factor_matrix, soil_fert_3D_spp_factor_matrix) * GDD_3D_spp_factor_matrix
        sprout_factor_matrix = ground_light_3D_spp_factor_matrix * partial_growth_factor_matrix

        #this also gets printed to local host terminal during model run for each year
        print "  Sprout Factor Matrix by Species: ", sprout_factor_matrix[0,0]
        print "Trees alive before planting = ", np.sum(DBH_matrix > 0.)
        # Plant new trees
        if driver['ALLOW_REGENERATION']:
            # plant some seedlings (roll the dice, see who gets picked)
            DBH_matrix, crown_base_matrix, seed_bank_matrix, \
            species_code_matrix, stress_flag_matrix = sprout_saplings_v2_3(DBH_matrix, crown_base_matrix,   #taken from ZELIG v2.3
                                                                           species_code_matrix, stress_flag_matrix, sprout_factor_matrix,
                                                                           seed_bank_matrix, basal_area_matrix)
            #print "Trees alive after planting  = ", np.sum(DBH_matrix > 0.)



        ### group by plot and by species
        #basal area computation
        basal_area_by_plot_by_species_matrix = compute_basal_area_by_plot_by_species(basal_area_matrix, species_code_matrix, number_of_species)
        #biomass computation
        biomass_by_plot_by_species_matrix = compute_biomass_by_plot_by_species(biomass_matrix, species_code_matrix, number_of_species)
        #stem density for tree with DBH>=8.0cm, grouped by species. This is computed on a plot by plot basis.
        stems_by_plot_by_species_matrix = compute_stems_by_plot_by_species(DBH_matrix, species_code_matrix, number_of_species)

        # store data at the beginning of each year
        # storing data : function in driver decides if data is stored this year (could be stored at 10-yr intervals, etc.)
        if driver['log_this_year_function'](year):
            simulation_years_logged.append(year)
            sim_year = '%.4d' % year
            DBH_group[sim_year] = DBH_matrix
            SpeciesCode_group[sim_year] = species_code_matrix
            BasalArea_group[sim_year] = basal_area_by_plot_by_species_matrix
            Biomass_group[sim_year] = biomass_by_plot_by_species_matrix
            DBH_distribution_group[sim_year] = stems_by_plot_by_species_matrix
            if driver['DEBUG']:
                # store the environmental conditions
                DegreeDays_group[sim_year] = GDD_matrix
                RelativeDryDays_group[sim_year] = drydays_fraction_mat
                # store the computed values that are intermediaries to the factor calculations
                RelativeSoilFertility_group[sim_year] = relative_soil_fertility_matrix                              # size: nx,ny
                # store the individual environmental factors for each tree (size: nx,ny,ntrees)
                GrowingDegreeDaysFactor_group[sim_year] = GDD_3D_spp_factor_matrix        # size: nx,ny,nspp
                SoilMoistureFactor_group[sim_year] = soil_moist_3D_spp_factor_matrix      # size: nx,ny,nspp
                SoilFertilityFactor_group[sim_year] = soil_fert_3D_spp_factor_matrix      # size: nx,ny,nspp
                AvailableLightFactor_group[sim_year] = available_light_spp_factor_matrix  # size: nx,ny,ntrees
                GrowthFactor_group[sim_year] = growth_factor_matrix                       # size: nx,ny,ntrees
                SproutFactor_group[sim_year] = sprout_factor_matrix                       # size: nx,ny,nspp

    ##close the climate record hdf file
    #if driver["CLIMATE_RANDOM"] == False:
    #    hdf_climate_file.close()

    # to the driver, add the list of years logged in the hdf file 
    driver['simulation_years_logged'] = simulation_years_logged
    # store the driver as a pickled (dill) string in the hdf file (this allows access to driver from HDF file upon analysis)
    hdf_file['driver'] = np.array(dill.dumps(driver))
    # store a string copy of the driver file in the hdf
    driver_str = np.array(open(driver_file).read())
    hdf_file['driver_as_string'] = driver_str

    # done with the hdf file for now, so close it
    hdf_file.close()
    for spp in range(number_of_species):
        print spp, species_code_to_name[spp]
    print "Simulation complete. Output file: %s" % (output_filename)





#Below are the functions called by the model flow above:

def initialization(driver_filename, output_filename, driver):
    """
    Generate the driver matrices that are only computed once (during the first year) and then used them throughout each year of sim:
        - get the plot size from the DEM
        - get the GIS-based monthly radiation matrices for PET calculation
        - get the GIS-based annual radiation matrix for terrain-shading fraction on each plot
    Generate the 4 matrices for annual model output storage at the individual-level.
    Create a tree species look up table.
    Ascertain plot size (=horizontal resolution for some parameters, such as soil moist/fert, and =vertical resolution for light subroutine).
    Generate the ray tracing directions (arrows) for direct and diffuse light computation.
    Prepare the available light matrix for light computation by incorporating DEM elevation information into it, so that the light 
    computation occurs above ground level.
    Generate a normalized seed list from the driver, which divides each spp's seeds by the total number of seeds, so that the sum =1.    

    Parameters: driver_filename -- the name of the driver file located in the same directory as the ZELIG-BORK code

    GIS drivers stored in driver file: radiation_mat_lst -- a list of 12 monthly accumulated radiation (WH/m^2) matrices the size of the 
                                                              simulation grid
                                       elevation_lapse_rate_adjustment_matrix -- 30x30 matrix, provides lapse-rate based adjustment to temperature          #Note, size same as DEM
                                                                                 from a 180m amls base
                                       site_index_mat - GIS-generated 30x30 matrix with plot-level site index (1-5), which will be converted to soil fert
                                       soil_fert_mat - 30x30 matrix of soil fert in Mg/ha/yr based on site index mat and a conversion table based on 
                                                       forestry yield tables
                                       permafrost_mat - 30x30 matrix of 0.0 for plots w/o permafrost and 1.0 for plots with permafrost

    Returns:    driver -- a driver "dictionary" that contains all of the driver information and values can be called by keys
                intial_soil_water_mat   --  initialized at field capacity (FC) to start the simulation off with sufficient 
                                            soil moisture
                tree_type_look_up_table --  a list of species participating in sim
	            arrows_list 			--  list of tuples for direct & diffuse angles for light computation (see manual)
                actual_leaf_area_mat    --  contains -1 below ground and 0 above ground for each plot and air space above plot
                                            size: nx, ny, vertical space = (max_tree_ht+(max elevation in sim - min elevation in sim))
                dem_offset_index_mat    --  a matrix with elevations above the minimum elev on sim'ed terrain, 
                                            size: simulation grid, used to elevate trees to local ground level on each plot
                radiation_fraction_mat  --  a matrix the size of the sim'ed grid that represents shading by terrain, and 
                                            contains values 0 to 1 for the amount of max possible light available at 
                                            ground level (same as at top-of-canopy)
                plot_area -- usually specified at 10m x 10m (horizontal resolution) by 1m (vertical resolution)
                DBH_matrix -- records DBH for every tree in sim, size: sim grid by number of trees on each plot
                crown_base_matrix   -- records the crown base for each tree in sim, size: sim grid by number of trees 
                                       on each plot
                species_code_matrix -- records the specie of each tree in sim, size: sim grid by number of trees on each plot
                stress_flag_matrix  -- records the stress flags for each tree in sim, size: sim grid by number of trees 
                                       on each plot
    """
    #size of simulation area, the size of each plot, and the North-West corner coordinate
    driver['DEM_mat'], x_size_m, y_size_m, NWx, NWy = read_in_ascii_attributes(driver['DEM_file_path'])
    # hardcode 1m as the step size in the vertical dimension (may want to change this later to accelerate light computation)
    z_size_m = 1
    nx,ny = driver['DEM_mat'].shape
    SQ_METERS_IN_HA = 10000.
    # toral area being simulated in hectares
    driver['sim_area_ha'] = (nx*x_size_m)*(ny*y_size_m)/SQ_METERS_IN_HA
    # number of plots in the East-West direction
    driver['EW_number_of_plots'] = nx
    # number of plots in the North-South direction
    driver['NS_number_of_plots'] = ny
    # side length of each plot in the East-West direction
    driver['EW_plot_length_m'] = x_size_m
    # side length of each plot in the North-South direction
    driver['NS_plot_length_m'] = y_size_m
    # simulated vertical step size in meters (related to light simulation)
    driver['vertical_step_m'] = z_size_m
    # area of each plot in m^2
    driver['plot_area_m2'] = x_size_m*y_size_m
    # maximum number of trees in this simulation
    driver['max_trees_in_simulation'] = nx * ny * driver['MAX_TREES_PER_PLOT']
    # the north-west corner coordinate as a tuple (units = meters, due to UTM projection of DEM)
    driver['north_west_corner_coordinates'] = (NWx, NWy)

    # for the growing degree day and soil moisture matrices:
    driver['elevation_lapse_rate_adjustment_matrix']=read_in_ascii(driver['elevation_lapse_rate_adjustment_matrix_filepath'])  #degrees C
    (plots_x, plots_y) = driver['elevation_lapse_rate_adjustment_matrix'].shape
    if (plots_x, plots_y) != (nx,ny):
        raise Exception("Lapse rate adjustment file wrong shape: %s" % driver['elevation_lapse_rate_adjustment_matrix_filepath'])
    initial_soil_water_mat = np.zeros(driver['elevation_lapse_rate_adjustment_matrix'].shape)
    initial_soil_water_mat = driver['FC'] #start sim with field capacity (FC) as soil water content (cm in the top meter of soil)
    # for PET, which is needed for soil moisture calculation:
    monthly_radiation_files_path = driver['monthly_radiation_files_path']  #WattHours/m2 accumulated for each month
    driver['radiation_mat_lst'] = one_time_radiation_readin(monthly_radiation_files_path,nx,ny)
    # for the soil fertility matrices:
    driver['site_index_mat'] = read_in_ascii(driver["Site_index_file_path"])  #values 1-5, 1=fertile soil, 5=poor soil, based on Orlov??? 1927 classification of soils for Russia
    if driver['site_index_mat'].shape != (nx,ny):
        raise Exception("Site index file wrong shape: %s" % driver['Site_index_file_path'])
    # site index to biovolume increment (m^3/plot)/yr
    driver['biovolume_soil_fert_mat'] = site_index_to_biovolume_convert(driver['site_index_mat'], driver['plot_area_m2'])  #Site index converted to a cap on gross primary productivity ((m3/ha)/yr)
    # for the permafrost matrices:
    driver['permafrost_mat'] = read_in_ascii(driver["permafrost_file_path"]) #Boolean 0 or 1 for presence of permafrost
    if driver['permafrost_mat'].shape != (nx,ny):
        raise Exception("Permafrost file wrong shape: %s" % driver['permafrost_file_path'])

    # initialize the simulation matrices
    if driver['LOAD_INTIIAL_CONDITIONS']:
        # load the simulation "state" from values in the driver to initialize model run from a known state with trees
        DBH_matrix = driver['INITIAL_CONDITIONS']['DEM_matrix'] #DBH in cm
        crown_base_matrix = driver['INITIAL_CONDITIONS']['CrownBase_matrix'] #height above ground in meters
        species_code_matrix = driver['INITIAL_CONDITIONS']['SpeciesCode_matrix']
        # Provide a good error message that the matrices in the driver are the wrong size
        if DBH_matrix.shape != (plots_x, plots_y, driver['MAX_TREES_PER_PLOT']) or \
           crown_base_matrix.shape != (plots_x, plots_y, driver['MAX_TREES_PER_PLOT']) or \
           species_code_matrix.shape != (plots_x, plots_y, driver['MAX_TREES_PER_PLOT']):
            raise Exception("Initial conditions incorrect in driver - matrices wrong shape")
    else:
        # initialize from bare ground (no trees)
        # initializing the 3-D arrays for storing DBH, crown base, species code, and stress flags for each tree on each plot
        DBH_matrix = np.zeros((plots_x,plots_y,driver["MAX_TREES_PER_PLOT"])) #(x,y,z) passed in as a tuple, DBH in cm
        crown_base_matrix = np.zeros((plots_x,plots_y,driver["MAX_TREES_PER_PLOT"]), dtype=np.int)  #height above ground in meters
        species_code_matrix = np.zeros((plots_x,plots_y,driver["MAX_TREES_PER_PLOT"]), dtype=np.int) #use a number to denote species code (1=BEPE, etc)
        species_code_matrix[:] = -1 #initialize empty as -1 so as not to confuse with 0 which corresponds to an actual species code in the simulation
    # stress flags always start at zero
    stress_flag_matrix = np.zeros((plots_x,plots_y,driver["MAX_TREES_PER_PLOT"]), dtype=np.int) #0=not stressed, >0 value corresponds to # of stress flags accrued
    

    #implement the inseeding lag for shade-tolerant species as in ZELIG v2.3
    #this allows shade intolerant species to sprout in earlier years of sim (yr 2, 3) than shade tolerant species (yr 5,6)
    MAX_LAG_YEARS = max(species_code_to_inseeding_lag) + 1
    species_in_sim = number_of_species
    seed_bank_matrix = np.zeros((MAX_LAG_YEARS,species_in_sim,nx,ny))  #cohorts(MY,MS,MR,MC) in ZELIG v2.3

    # Designate whether using 3-dimensional light ray tracing (direct light from 7 compass directions and 7 sun elev angles)
    # or 1-dimensional light from just overhead used for independent plot mode (all light is diffuse).
    # If the wrong mode is specified, prints out an error.
    if driver['LIGHT_MODE']=='3D':
        # for the 3D light subroutine:
        # define the plot size along the x,y, and z dimentions: taken from the DEM, and 1-m vertical step is assumed (hardcoded)
        # pre-compute the "arrows" that will be used to compute the light at every position in the actual leaf area matrix
        arrows_list = build_arrows_list(xsize=x_size_m, ysize=y_size_m, zsize=z_size_m, PHIB=driver['PHIB'], PHID=driver['PHID'])
    elif driver['LIGHT_MODE']=='1D':
        arrows_list = build_arrows_list_independent(xsize=x_size_m, ysize=y_size_m, zsize=z_size_m)  #no need for PHIB or PHID b/c from overhead all light is diffuse
        #print "1-D arrows list:", arrows_list
        #assert(False)
    else:
        raise Exception("Can't grow trees in hyperbolic space! Select a proper LIGHT_MODE in driver (1D or 3D)!")
    # set up an empty actual leaf area matrix that has been adjusted for elevation in the DEM = these are the initial conditions w/o tree but w/terrain
    actual_leaf_area_mat, dem_offset_index_mat = prepare_actual_leaf_area_mat_from_dem(dem_mat=driver['DEM_mat'], 
                                                             zsize=z_size_m, 
                                                             max_tree_ht=driver['MAX_TREE_HEIGHT'])  # the maximum height a tree can reach in m
    # Read in the radiation fraction matrix that is used to scale the proportion of radiation (unitless ratio)
    # that is available to every location on the DEM at ground-level. This will be used to account for terrain shading.
    # Ground-level light, post accounting for shading by terrain, is pretty much the same as the
    # fraction used for top-of-canopy, since found no significant difference b/w the two.
    radiation_fraction_mat = np.array(read_in_ascii(filename=driver["Radiation_fraction_file_path"]), dtype=np.float) #unitless
    if radiation_fraction_mat.shape != (nx,ny):
        raise Exception("Rad fraction file wrong shape: %s" % driver['Radiation_fraction_file_path'])
    # set the plot size
    plot_area=driver['plot_area_m2']

    # associate a species name/acronym to each species code look up table in the driver
    driver['name_to_species_code'] = name_to_species_code
    driver['species_code_to_name'] = species_code_to_name

    # initialize HDF file in which annual model output will be stored
    hdf_file = h5py.File(output_filename, "w")

    #hdf_file.attrs["driver"] = ns
    grp1 = hdf_file.create_group("DBH")
    grp2 = hdf_file.create_group("SpeciesCode")
    grp3 = hdf_file.create_group("BasalArea")
    grp4 = hdf_file.create_group("Biomass")
    grp5 = hdf_file.create_group("DBH_distribution")
    # initialize groups for debugging
    if driver['DEBUG']:
        # weather
        grp6 = hdf_file.create_group("DegreeDays")
        grp7 = hdf_file.create_group("RelativeDryDays")
        # intermediate calculations
        grp8 = hdf_file.create_group("RelativeSoilFertility")    # size: nx,ny
        # factors
        grp9 = hdf_file.create_group("GrowingDegreeDaysFactor")  # size: nx,ny,nspp
        grp10 = hdf_file.create_group("SoilMoistureFactor")      # size: nx,ny,nspp
        grp11 = hdf_file.create_group("SoilFertilityFactor")     # size: nx,ny,nspp
        grp12 = hdf_file.create_group("AvailableLightFactor")    # size: nx,ny,ntrees
        grp13 = hdf_file.create_group("SproutFactor")            # size: nx,ny,nspp
        grp14 = hdf_file.create_group("GrowthFactor")            # size: nx,ny,ntrees


    return driver, initial_soil_water_mat, \
           arrows_list, actual_leaf_area_mat, dem_offset_index_mat, radiation_fraction_mat, plot_area, \
           DBH_matrix, crown_base_matrix, species_code_matrix, stress_flag_matrix, seed_bank_matrix, hdf_file



def generate_weather(driver, initial_soil_water_mat, year):
    """
    Generate average monthly temperatures and precipitation sums from avg and std values provided in driver file from 
    historical data.

    Parameters: driver -- a driver "dictionary" that contains all of the driver information and values can be called by keys
                intial_soil_water_mat -- initialized at field capacity (FC) to start the simulation off with sufficient
                                         soil moisture (cm in top 1m of soil)
                year -- current simulation year (integer)

    Returns:    soil_moisture_mat --     a fraction 0 to 1 of soil moisture on each plot based on precip sim and PET calc;
                                         PET is temp & readiation-based
                GDD_matrix --            a matrix the size of the sim grid of accumulated growing degrees over this year in sim
                drydays_fraction_mat -- the fraction 0 to 1 of the growing season in drought
    """
    temperature_avg_vec, temperature_std_vec, rain_avg_vec, rain_std_vec = driver['return_annual_weather_function'](year)  #units: deg C, deg C, cm, cm

    GDD_matrix, \
    monthly_temp_mat_lst, \
    total_growing_season_mat = GrowingDegreeDays_calc(ddbase = driver["DDBASE"], monthly_temperature_avgs_lst = temperature_avg_vec, 
                                                      monthly_temperature_stds_lst = temperature_std_vec, 
                                                      monthly_temperature_mins_lst = driver["minT"],
                                                      monthly_temperature_maxs_lst = driver["maxT"],
                                                      lapse_rate_adj_mat = driver['elevation_lapse_rate_adjustment_matrix'])
    # simulate random precipitation
    monthly_sim_rain_vec = rain_sim(rainfall_monthly_avgs = rain_avg_vec, rainfall_monthly_stds = rain_std_vec)
    
    radiation_mat_lst = driver["radiation_mat_lst"]  #Watt-Hours/m2, a matrix for each month contains a montly value for each plot
    soil_moisture_mat, \
    drydays_fraction_mat = drydays(total_growing_season_mat = total_growing_season_mat, soil_moisture_mat = initial_soil_water_mat, 
                                   wilting_point = driver['WP'], monthly_temp_mat_lst = monthly_temp_mat_lst, 
                                   radiation_mat_lst = radiation_mat_lst, field_capacity = driver['FC'],
                                   monthly_sim_rain_vec = monthly_sim_rain_vec, ddbase = driver["DDBASE"]) 

    return monthly_temp_mat_lst, soil_moisture_mat, GDD_matrix, drydays_fraction_mat


def site_index_to_biovolume_convert(site_index_mat, plot_area):
    """
    Convert site index to soil fertility in (m^3/ha)/yr. In actuality, there is not a known clear relationship
    between site index and quantitative amounts of "nutrition" (i.e. nitrogen) in the soil. Site index (value 1-5) is a species-
    specific value assigned to a parcel of land based on how fast trees of a given species grow on it. The way soil 
    foresters do this is a bit of "black magic", as Hank calls it. Possibly due to scale incompatibilities, there is no
    clear deliniation or overlap of {soil types and nitrogen content} with site indices in the Usolsky forest inventory.
    Here, the conversion is derived from yield tables and is species-specific, however, most of the 7 test species agree
    on ~4m3/ha as the max annual biovolume increment for site index (SI) 3, and other SIs are similar. Adjust the conversion
    dictionary based on monospecies/mixed forest simulation, stocking density, etc. Useful tables for stocking density
    adjustments in back of yield table book p.726 onward.

    Parameters:  site_index_mat -- output from GIS showing 1 value of site index (1-5) for each plot in sim
                                   1 is best soil; 5 is worst soil; these values can also be user-specified

    Returns:     soil_fert_mat -- limit on annual biovolume accretion in (m^3/plot)/yr for each plot
    """
    #site_index_look_up_table = {1:13.78, 2:10.18, 3:8.45, 4:6.7, 5:4.96}  # original in (m^3/ha)/yr; this is based on yield table production
    #site_index_look_up_table = {1:13.78, 2:10.18, 3:8.45, 4:6.7, 5:4.96}  # original in (m^3/ha)/yr; this is based on yield table production
    #site_index_look_up_table = {1:9.07, 2:3.96, 3:2.92, 4:1.99, 5:1.27}  # original in (m^3/ha)/yr; this is based on yield table production LASI p.231
    #site_index_look_up_table = {1:8.0, 2:6.35, 3:2.5, 4:1.99, 5:1.3}  # PISY p.136-137 for SI 3:1.3 as on p.728
    site_index_look_up_table = {1:8.0, 2:6.0, 3:4.0, 4:2.0, 5:1.3}  #for treeline sim = 5.0 **change SI 3 back to 7.0 for ABSI/BEPE
    soil_fert_mat = np.zeros(site_index_mat.shape)
    nx,ny = soil_fert_mat.shape
    for x in range(nx):
        for y in range(ny):
            soil_fert_mat[x,y] = site_index_look_up_table[site_index_mat[x,y]]   #(m3/ha)/yr
    # to convert from m^3/ha to m^3/plot:  1ha/10000m2 * X m2/plot
    m3_ha_to_m3_plot_conversion_factor = 1./10000. * plot_area  #no magic numbers
    soil_fert_mat = soil_fert_mat * m3_ha_to_m3_plot_conversion_factor  
    return soil_fert_mat

'''
##### soil fertility computed from biomass #####
def biomass_compute_relative_soil_fertility_matrix(DBH_matrix, biomass_matrix, optimal_growth_increment_matrix, optimal_biomass_matrix, soil_fert_mat):
    """
    don't actually use this one, b/c it's more error-prone than site-index-to-biovolume convert
    """
    nx,ny = soil_fert_mat.shape
    relative_soil_fertility_matrix = np.zeros((nx,ny))
    for x in range(nx):
        for y in range(ny):
            rel_fertility = biomass_plot_wide_relative_soil_fertility(DBH_matrix[x,y], biomass_matrix[x,y], 
                                                                      optimal_growth_increment_matrix[x,y], 
                                                                      optimal_biomass_matrix[x,y], soil_fert_mat[x,y])
            relative_soil_fertility_matrix[x,y] = rel_fertility

    return relative_soil_fertility_matrix


@numba.jit(nopython=True)
def biomass_plot_wide_relative_soil_fertility(DBH_vec, biomass_vec, opt_inc_vec, optimal_biomass_vec, soil_fertility):
    """
    Compute the relative soil fertility (0 to 1) as a ratio of
    the plot soil fertility (set in the driver and then normalized
    to the amount of biomass that could be grown on this plot under
    optimal conditons).

    Parameters: DBH_vec -- dbh of each tree on this plot
                biomass_vec -- current biomass (kg) of each tree on this plot
                opt_inc_vec -- the optimal growth increment of each tree on this plot
                optimal_biomass_vec -- the biomass (kg) of each tree on this plot under optimal growth conditions
                soil_fertility -- the soil fertility value (kg/ha) for this plot

    Returns: rel_fertility -- the relative soil fertility which is basically the percentage of
                              total optimal growth that the soil could support
    """
    
    optimal_biomass_increment = 0.
    for ind in range(len(DBH_vec)):
        dbh = DBH_vec[ind]
        if dbh > 0:
            # the biomass of the tree currently (in KG)
            current_biomass = biomass_vec[ind]
            # the additional biomass that could be grown under optimal conditions is the biomass
            # of the tree after optimal growth minus the current biomass
            optimal_diameter = dbh + opt_inc_vec[ind]
            # the optimal biomass increment is the additional biomass added when the dbh is optimally grown
            optimal_biomass = optimal_biomass_vec[ind]
#            assert optimal_diameter > dbh 
            optimal_additonal_growth_biomass = optimal_biomass - current_biomass
#            if optimal_biomass <= current_biomass:
#                print "dbh = ", dbh, "opt_D = ", optimal_diameter, "current biomass = ", current_biomass, "optimal biomass = ", optimal_biomass, "specie = ", tree.name
#            assert optimal_biomass > current_biomass
#            assert optimal_additonal_growth_biomass > 0.0
            optimal_biomass_increment += optimal_additonal_growth_biomass  #accumulated the optimal biomass increment for each tree of each species

    # compute the relative fertility (0 to 1) as the ratio of the biomass that the plot can grow in a year
    # to the amount of biomass that can be grown under optimal conditions
    # soil fertility (in KG/plot area) / optimal biomass increment (in KG/this plot area)
    if soil_fertility > optimal_biomass_increment:
        # there is more fertility available than can be used, soil fertility is not limiting
        rel_fertility = 1.0
    else:
        # there could be more demand for fertility than the soil can support, so throttle back by species-specific tolerances
        # both values in kg/plot
        rel_fertility = soil_fertility / optimal_biomass_increment

    return rel_fertility

##### end soil fertility computed from biomass #####
'''

############ biovolume increment using only above ground factors ################
@numba.jit()
def compute_plot_likely_biovolume_increment(species_code_matrix, GDD_3D_spp_factor_matrix, 
                                            available_light_spp_factor_matrix, 
                                            optimal_biovolume_increment_matrix):
    """
    Compute the biovolume that each plot will contain if the growth is not limited by any below ground factors such as soil moisture
    or soil fertility.

    Parameters: species_code_matrix -- the species code for every tree in the simulation
                              size : nx, ny, ntrees
                GDD_3D_spp_factor_matrix -- the growing degree day factor for every plot and every species
                              size : nx, ny, nspp
                available_light_spp_factor_matrix -- the light factor for every tree in the simulation
                              size : nx, ny, ntrees
                optimal_biovolume_increment_matrix -- the optimal biovolume increment that could occur under optimal conditions
                                                      for each tree in the simulation
                              size : nx, ny, ntrees

    Returns : likely_biovolume_increment_matrix -- the biovolume increment for each tree when only taking into account above
                                                   ground factors
                                                   size : nx,ny,ntrees
    """
    nx, ny, ntrees = species_code_matrix.shape
    likely_biovolume_increment_matrix = np.zeros((nx,ny,ntrees))

    for x in range(nx):
        for y in range(ny):
            for tree in range(ntrees):
                species_code = species_code_matrix[x,y,tree]
                degree_day_factor = GDD_3D_spp_factor_matrix[x,y,species_code]
                light_factor = available_light_spp_factor_matrix[x,y,tree]

                optimal_biovolume_increment = optimal_biovolume_increment_matrix[x,y,tree]
                likely_biovolume_increment_matrix[x,y,tree] = optimal_biovolume_increment * (degree_day_factor * light_factor)

    return likely_biovolume_increment_matrix   #units: m3 (by individual tree)
############ end biovolume increment using only above ground factors ############


##### soil fertility computed from biovolume #####
@numba.jit()
def biovolume_compute_relative_soil_fertility_matrix(likely_biovolume_increment_matrix,
                                                     plot_max_biovolume_increment_matrix): #comes from site index to biovolume convert
    """
    Compute the ratio of likely biovolume increment to what the plot can support.

    Parameters : likely_biovolume_increment_matrix -- the likely biovolume increment for each tree in the simulation
                                                      size : nx,ny,ntrees
                 plot_max_biovolume_increment_matrix -- the maximum biovolume increment that each plot can support (Gross Primary Productivity cap based on site index)
                                                      size : nx,ny

    Returns : relative_soil_fertility_matrix -- the ratio of what the plot can support to the possible biovolume increment
                                                size : nx,ny
    """
    nx,ny,ntrees = likely_biovolume_increment_matrix.shape
    relative_soil_fertility_matrix = np.zeros((nx,ny))
    # sum the biovolume increment for all trees on each plot, size:nx,ny
    plot_likely_biovolume_increment_mat = np.sum(likely_biovolume_increment_matrix, axis=2)

    for x in range(nx):
        for y in range(ny):
            plot_likely_biovolume_increment = plot_likely_biovolume_increment_mat[x,y]
            plot_max_biovolume_increment = plot_max_biovolume_increment_matrix[x,y]
            if plot_likely_biovolume_increment <= plot_max_biovolume_increment:
                relative_soil_fertility_matrix[x,y] = 1.0
            else:
                relative_soil_fertility_matrix[x,y] = plot_max_biovolume_increment / plot_likely_biovolume_increment  #this ratio is <1

    return relative_soil_fertility_matrix

##### end soil fertility computed from biovolume #####


@numba.jit()
def light_factor_compute(available_light_mat, tree_height_matrix, crown_base_matrix, species_code_matrix, number_of_species):
    """
    compute available light for each step through the canopy
    compute ground-level light

    Parameters: available_light_mat  --  value 0 to 1 representing the fraction of available light at each point in the 3-D grid
                                         of the simulation (think: volume slicer)
                tree_height_matrix   -- records the height for each tree in sim, size: sim grid by number of trees 
                                        on each plot
                crown_base_matrix    -- records the crown base for each tree in sim, size: sim grid by number of trees 
                                        on each plot
                species_code_matrix  -- records the specie of each tree in sim, size: sim grid by number of trees on each plot
                number_of_species    -- the number of species participating in sim

    Returns:    ground_light_spp_factor_matrix -- size: nx,ny,spp_in_sim
                available_light_spp_factor_matrix -- size: nx,ny,ntrees

    """
    nx, ny, ntrees = tree_height_matrix.shape
    available_light_spp_factor_matrix = np.zeros((nx,ny,ntrees))
    ground_light_spp_factor_matrix = np.zeros((nx,ny,number_of_species))
    # compute a 4D matrix (nx,ny,nz,spp) containing the light factors by height and by species
    light_factor_by_species_matrix = compute_available_light_factors_by_species(available_light_mat, number_of_species)

    # for each individual tree the light factor is averaged over the length of the tree crown (crown base to height)
    for x in range(nx):
        for y in range(ny):
            for ind in range(ntrees):
                height = tree_height_matrix[x,y,ind]
                crown_base = crown_base_matrix[x,y,ind]
                spp = species_code_matrix[x,y,ind]
                crown_length = numba.int_(height) - crown_base
                if spp >= 0 and crown_length > 0:
                    assert (numba.int_(height) > crown_base)
                    # average over the length of the tree crown
                    tree_light_factor = 0.
                    for z in range(crown_base, int(height)):
                        tree_light_factor += light_factor_by_species_matrix[x,y,z,spp]

                    tree_light_factor /= float(crown_length)
                    available_light_spp_factor_matrix[x,y,ind] = tree_light_factor

            # build the nx,ny,nspp ground_light_matrix used for sprouting
            for spp in range(number_of_species):
                ground_light_spp_factor_matrix[x,y,spp] = light_factor_by_species_matrix[x,y,0,spp]

    return ground_light_spp_factor_matrix, available_light_spp_factor_matrix


def sprout_saplings_v2_3(DBH_matrix, crown_base_matrix, species_code_matrix, stress_flag_matrix, 
                         sprout_factor_matrix, seed_bank_matrix, basal_area_matrix):
    """
    This function somewhat resembles ZELIG v2.3 REGEN subroutine, which includes stump sprouting and seedling sprouts.
    SIBBORK doesn't currently have stump sprouting, but additing this would increase the contribution from aspen, birch and spruce,
    especially post fire.

    Parameters:  DBH_matrix -- records DBH for every tree in sim, size: sim grid (x,y) by number of trees on each plot
                 crown_base_matrix -- records the crown base for each tree in sim, size: sim grid by number of trees 
                                      on each plot
                 species_code_matrix -- records the specie of each tree in sim, size: sim grid by number of trees on each plot
                 stress_flag_matrix -- records the stress flags for each tree in sim, size: sim grid by number of trees 
                                       on each plot
                 sprout_factor_matrix -- ground light * min(soil moist, soil fert) * GDD computed for each spp on plot
                                         size: sim grid by spp_in_sim
                 basal_area_matrix -- basal area (m^2) for each tree in the simulation
                                      size: nx, ny, ntrees

    Returns:     DBH_matrix -- records DBH for every tree in sim, size: sim grid by number of trees on each plot
                 crown_base_matrix -- records the crown base for each tree in sim, size: sim grid by number of trees 
                                      on each plot
                 species_code_matrix -- records the specie of each tree in sim, size: sim grid by number of trees on each plot
                 stress_flag_matrix -- records the stress flags for each tree in sim, size: sim grid by number of trees 
                                       on each plot
    """
    #set onion shape:
    nx,ny,ntrees = DBH_matrix.shape
    species_in_sim = number_of_species

    #peel the onion:
    for x in range(nx):  #for each row
        for y in range(ny):  #for each column
            #total_transplants = 0   #this year's seed-bank-to-saplings migrants for all species on this plot
            transplants_by_species = []  #in preparation for the seed dart board; seed value for each species that will be planted this year
            for species_code in range(number_of_species):
                if sprout_factor_matrix[x,y,species_code] == 0.:  #if cannot regenerate because regen factor is 0,
                    seed_bank_matrix[:,species_code,x,y] = 0   #erase seeds of this species from seed bank
                else: 
                    seed_bank_matrix[1:,species_code,x,y] = seed_bank_matrix[0:-1,species_code,x,y]  #shift lag year by 1 yr to more recent
                #place seeds in the 1st year of seed bank based on the regen factor for each species
                seed_bank_matrix[0,species_code,x,y] = species_code_to_SEED[species_code] * sprout_factor_matrix[x,y,species_code]

                #keep track of seeds to be moved from seed bank to saplings
                species_lag = species_code_to_inseeding_lag[species_code]
                transplants = seed_bank_matrix[species_lag,species_code,x,y]   #returns seed bank for this species species_lag years ago
                #total_transplants += transplants   #accumulate transplants across all species for this plot this year
                transplants_by_species.append(transplants)  #append seeds for each species that are eligible for saplinghood this year

            if sum(transplants_by_species)!=0:  #making sure not to divide by 0 on the first year of sim; no need to compute all that stuff below when there are no trees in sim
                trees_on_plot = sum([1 for DBH in DBH_matrix[x,y] if DBH>0.0]) #count the number of trees on this plot
                total_basal_area_on_plot = np.sum(basal_area_matrix[x,y]) #sum the basal area on this plot
                #Compute number of available spaces for new seedlings or stump sprouts on this plot based on number of
                #trees on plot and the total basal area they occupy (on this plot).
                #Not sure how subtracting individuals and area from individuals makes sense: subtracting apples from lobsters? retained from ZELIG. never fill plot to max # trees.
                possible_spaces = int(ntrees - trees_on_plot - total_basal_area_on_plot) #scales appropriately with plot size
                normalized_transplant_vec = np.array(transplants_by_species)/(sum(transplants_by_species)) #normalizing seeds as in v2.3
                cumsum_transplant_vec = np.cumsum(normalized_transplant_vec) #prepare the dartboard
                saplings_to_plant_by_species = np.zeros(species_in_sim, dtype=np.int)  #create bins for each species to count number of seeds in each

                for i in range(possible_spaces):
                    seed_dart = random.random()  #random 0 to 1 number, includes 0 but not 1, i.e. [0,1)
                    for species_code in range(species_in_sim):
                        if seed_dart <= cumsum_transplant_vec[species_code]:  #throw the dart, make a selection
                            saplings_to_plant_by_species[species_code]+=1  #increment seeds in the appropriate species bin
                            break

                #plant the seeds selected for transplant as seedlings
                individual = 0
                for species_code in range(species_in_sim):
                    ntransplants = saplings_to_plant_by_species[species_code]  # number of transplants by species
                    for n in range(ntransplants):
                        #for individual in range(ntrees): #address of each tree in each cell within the 3-D matrix
                        while individual < ntrees:  #fill in the gaps
                            if not (DBH_matrix[x,y,individual] > 0.0):  #only plant if there's no tree there already
                                # plant a new sapling of this tree type
                                # update DBH_matrix with DBH (cm) of new sapling (initial DBH hardcoded)
                                DBH_matrix[x,y,individual] = 2.5 + 0.25 * random.gauss(mu=0.0,       # average
                                                                                       sigma=1.0)    # standard deviation
                                # update species_code_matrix with integer number corresponding to the species of the new sapling
                                species_code_matrix[x,y,individual] = species_code
                                # set crown_base_matrix to 0
                                crown_base_matrix[x,y,individual] = 0.0
                                # set stress_flag_matrix to 0
                                stress_flag_matrix[x,y,individual] = 0.0    
                                individual += 1
                                break  #so as not to plant all saplings on top of each other; break exits the while loop
                            else:
                                individual += 1 #increment individual, so don't have to restart at index 0 in the search for empty spaces
 
    return DBH_matrix, crown_base_matrix, seed_bank_matrix, species_code_matrix, stress_flag_matrix



#@numba.jit(nopython=True)
def grow_trees(DBH_matrix, species_code_matrix, available_light_spp_factor_matrix, 
                     partial_growth_factor_matrix, stress_flag_matrix, optimal_growth_increment_matrix,species_code_to_stress_threshold):
    """
    Increment DBH based on optimal diameter increment (passed in) and the stresses the tree experiences based on environmental conditions/limitations.

    Parameters:  DBH_matrix -- records DBH for every tree in sim, size: nx,ny,ntrees 
                 species_code_matrix -- records the specie of each tree in sim, size: nx,ny,ntrees 
                 available_light_spp_factor_matrix -- matrix size: nx,ny,ntrees
                 partial_growth_factor_matrix -- matrix size: nx,ny,spp_in_sim
                 stress_flag_matrix -- matrix size: nx,ny,spp_in_sim
                 optimal_growth_increment_matrix -- optimal growth increment (cm) for each tree
                                                    matrix size: nx,ny,spp_in_sim
                 species_code_to_stress_threshold -- look-up-table for species-specific minimum annual growth requirements (as fraction of opt inc)

    Returns:    DBH_matrix -- records DBH for every tree in sim, size: sim grid by number of trees on each plot
                stress_flag_matrix -- the stress flag accumulator for each tree in the simulation; size: nx,ny,ntrees
                growth_factor_matrix -- the growth factor for each tree in the simulation; size: nx,ny,ntrees
    """
    nx,ny,ntrees = DBH_matrix.shape
    growth_factor_matrix = np.zeros((nx,ny,ntrees))
    for x in range(nx):
        for y in range(ny):
            for individual in range(ntrees): #address of each cell in 3-D matrix
                tree_DBH = DBH_matrix[x,y,individual]   #pulling out the DBH (in cm) of each individual
                if tree_DBH > 0:
                    species_code = species_code_matrix[x,y,individual] #pulling out the species code for each individual
                    optimal_growth_increment = optimal_growth_increment_matrix[x,y,individual]
                    light_factor_for_tree = available_light_spp_factor_matrix[x,y,individual] #pull out the available light in a pillar at each tree's location
                    soil_and_temp_factor_this_plot = partial_growth_factor_matrix[x,y,species_code] #this is the partial growth factor (min(soilmoist,soilfert,permafrost)*GDD)
                    growth_factor = light_factor_for_tree * soil_and_temp_factor_this_plot #compute whole growth factor: light* min(soilmoist,soilfert) *GDD); need to do it like this b/c the matrices for the environmental factors are different sizes
                    actual_growth_increment = optimal_growth_increment * growth_factor #scale down the optimal growth increment by resource limitations
                    #if actual_growth_increment <= 0.1*optimal_growth_increment: #same as if growth_factor<0.1
                    if actual_growth_increment <= species_code_to_stress_threshold[species_code]*optimal_growth_increment: #same as if growth_factor<species_code_to_stress_threshold
                        stress_flag_matrix[x,y,individual] = stress_flag_matrix[x,y,individual] + 1
                    else:
                        stress_flag_matrix[x,y,individual] = 0   #stress flags erased in a good growth year ("release")
                    new_tree_DBH = tree_DBH + actual_growth_increment #compute the new DBH
                    DBH_matrix[x,y,individual] =  new_tree_DBH #assign than new DBH to storage matrix
                    # store the growth factor for this tree (for debug purposes)
                    growth_factor_matrix[x,y,individual] = growth_factor

    return DBH_matrix, stress_flag_matrix, growth_factor_matrix


def kill_trees(DBH_matrix, species_code_matrix, stress_flag_matrix, crown_base_matrix, number_of_species):
    """
    

    Parameters:  DBH_matrix -- records DBH for every tree in sim, size: sim grid by number of trees on each plot
                 species_code_matrix -- records the specie of each tree in sim, size: sim grid by number of trees on each plot
                 stress_flag_matrix -- records the stress flags for each tree in sim, size: sim grid by number of trees 
                                       on each plot
                 crown_base_matrix -- records the crown base for each tree in sim, size: sim grid by number of trees 
                                      on each plot
                 number_of_species -- scalar number of species in this simulation

    Returns:      DBH_matrix -- records DBH for every tree in sim, size: sim grid by number of trees on each plot
                  stress_flag_matrix -- records the stress flags for each tree in sim, size: sim grid by number of trees 
                                        on each plot
    """
    # compute a matrix containing the age mortality probability for each tree
    age_mortality_probability_mat = np.zeros(DBH_matrix.shape, dtype=np.double)
    for current_species_code in range(number_of_species):
        mat_locations = (species_code_matrix == current_species_code)
        age_mortality_probability_mat[mat_locations] = species_code_to_age_mortality_function[current_species_code]()  #species-specific through AGEMAX
    # compute a matrix of random numbers
    a_random_mat = np.random.random(DBH_matrix.shape)
    b_random_mat = np.random.random(DBH_matrix.shape)

    return kill_trees_numba(DBH_matrix, species_code_matrix, stress_flag_matrix, crown_base_matrix, 
                            age_mortality_probability_mat, a_random_mat, b_random_mat)


@numba.jit(nopython=True)
def kill_trees_numba(DBH_matrix, species_code_matrix, stress_flag_matrix, crown_base_matrix, 
                     age_mortality_probability_mat, a_random_mat, b_random_mat):
    nx,ny,nz = stress_flag_matrix.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz): #address of each cell in 3-D stress flag matrix
                # STRESS-RELATED MORTALITY
                flag = stress_flag_matrix[x,y,z]
                if flag >=2:  #if a tree has accumulated 2 stress flags by growing less than minimum requirement for 2 consecutive years
                    if a_random_mat[x,y,z] <= 0.37:   #throw the dart, tree will die due to stress (stress mortality); 1% chance of surviving 10 stressed year, retained from ZELIG & Shugart, 1984.
                        stress_flag_matrix[x,y,z] = 0 # stress flags stored as integers
                        DBH_matrix[x,y,z] = 0.0 # kill tree, DBH values stored as floats
                        species_code_matrix[x,y,z] = -1 #reset species code to empty
                        # set crown_base_matrix to 0
                        crown_base_matrix[x,y,z] = 0.0
 
                # RANDOM (AGE-REALTED) MORTALITY
                # get a uniform random number between 0 and 1 and see if it falls within the probability of natural mortality
                if b_random_mat[x,y,z] <= age_mortality_probability_mat[x,y,z]:
                    stress_flag_matrix[x,y,z] = 0 # stress flags stored as integers
                    DBH_matrix[x,y,z] = 0.0 # DBH values stored as floats
                    species_code_matrix[x,y,z] = -1 #reset species code to empty
                    # set crown_base_matrix to 0
                    crown_base_matrix[x,y,z] = 0.0
 
                # OVER-PRUNE MORTALITY
                #if crown length = 0, kill the tree, but actually in Avail Light Growth Factor (ALGF) computation, if crown length=0, ALGF=0, so flagged for stress mortality anyway
                          
    return DBH_matrix, stress_flag_matrix, species_code_matrix, crown_base_matrix


def compute_light(actual_leaf_area_mat, dem_offset_index_mat, radiation_fraction_mat, arrows_list, xsize,ysize,zsize):
    """

    Parameters:  actual_leaf_area_mat   --  contains the column of actual leaf area for each plot in the simulation grid
                                            size: nx, ny, max_tree_ht
                 dem_offset_index_mat   --  contains an index that helps to track the relative elevation of the DEM
                                            under each plot in the simulation grid
                 radiation_fraction_mat --  a matrix the size of the sim'ed grid that represents shading by terrain, and 
                                            contains values 0 to 1 for the amount of max possible light available at 
                                            ground level (same as at top-of-canopy)
	             arrows_list 			--  list of tuples for direct & diffuse angles for light computation (see manual)
                 xsize,ysize,zsize      --  dimensions along the edges of each plot
                                            usually specified at 10m x 10m (horizontal resolution) 
                                            by 1m (vertical resolution)
                 
    Returns:     available_light_mat -- available light matrix
    """
    #setup the actual_leaf_area_mat shape based on new nz, which includes terrain and max tree height "head space"
    nx,ny,nz = actual_leaf_area_mat.shape
    # finally compute the available light matrix based on the actual leaf area and the "arrow" directions and proportions
    # actual_leaf_area_mat will have 0s and -1s in it during 1st year of sim, then populated with other values (-1s stay same)
    available_light_mat = compute_3D_light_matrix(actual_leaf_area_mat, 
                                                  dem_offset_index_mat,
                                                  radiation_fraction_mat, #scaled by max radiation received on s-facing slopes >10deg, computed in GIS
                                                  xsize,ysize,zsize,  # dimesions (in meters) along each plot box
                                                  arrows_list) #list of tuples for direct & diffuse angles for light computation

    return available_light_mat


@numba.jit
def compute_crown_base(DBH_matrix, species_code_matrix, tree_height_matrix, 
                         crown_base_matrix, available_light_mat):
    """
    Using the available light and species-specific compensation points (based on shade-tolerance class) compute the height
    of base of crown based on where foliage no longer receives enough light to grow.

    Parameters:  DBH_matrix              --  records DBH for every tree in sim, size: sim grid by number of trees on each plot
                 species_code_matrix     --  records the specie of each tree in sim, size: sim grid by number of trees on each plot
                 height_matrix           --  the height of each tree in meters
                 crown_base_matrix       --  records the crown base in meters for each tree in sim, 
                                             size: sim grid by number of trees on each plot
                 available_light_mat     --  value 0 to 1 representing the fraction of available light at each point in the 3-D grid
                                             of the simulation (think: volume slicer)
                                             size: nx, ny, vertical space = (max_tree_ht+(max elevation in sim - min elevation in sim))

    Returns:     crown_base_matrix       --  records the crown base in meters for each tree in sim, 
                                             size: sim grid by number of trees 
    """
    # get a list of the compensation points by tree type
    CP_lookup_table = np.array(species_code_to_CP)

    nx,ny,ntrees = DBH_matrix.shape
    for x in range(nx):
        for y in range(ny):
            for individual in range(ntrees): #address of each cell in 3-D matrix
                species_code = species_code_matrix[x,y,individual] #get species code from matrix
                tree_dbh = DBH_matrix[x,y,individual] #get tree DBH from DBH matrix
                tree_height = tree_height_matrix[x,y,individual]
                crown_base = int(crown_base_matrix[x,y,individual]) #should already be an integer
                crown_base_matrix[x,y,individual] = crown_base

    return crown_base_matrix


def compute_basal_area_by_plot_by_species(basal_area_matrix, species_code_matrix, number_of_species):
    """
    Computes basal area for each tree in the DBH matrix based on the tree's DBH. 
    The equation is the same for all species. The plumbing is in place to change
    this in Tree.py, w/o having to change the rest of the code, if want to use
    species-specific basal area calculations. Using a pandas dataframe, compute 
    basal area for each tree in the DBH matrix based on the tree's DBH and 
    species-specific basal area equation. Group basal areas by plot and by species. 
    Get rid of the -1 species (-1 is a flag for no tree/no species). 
    Reshape pandas dataframe into 
    basal area matrix, which now contains only the plot by plot sum of basal area 
    values for each species present on the plot.


    Parameters:  basal_area_matrix         --  records basal area for every tree in sim, size: sim grid by number of trees on each plot
                 species_code_matrix       -- record the species code for every tree in sim; size: sim grid by number of trees on each plot
                 number_of_species         -- integer number of species in this simulation

    Returns:     grouped_basal_area_matrix  --  returns a basal area (in square meters) summed up for every species on a plot
                                        size: nx,ny,spp_in_sim
    """
    nx, ny, nind = basal_area_matrix.shape
    grouped_basal_area_matrix = np.zeros((nx,ny,number_of_species))
    return compute_basal_area_by_plot_by_species_numba(basal_area_matrix, species_code_matrix, number_of_species, grouped_basal_area_matrix)

@numba.jit(nopython=True)
def compute_basal_area_by_plot_by_species_numba(basal_area_matrix, species_code_matrix, number_of_species, grouped_basal_area_matrix):
    nx, ny, nind = basal_area_matrix.shape

    for x in range(nx):
        for y in range(ny):
            for spp in range(number_of_species):
                basal_sum = 0.
                for ind in range(nind):
                    species_code = species_code_matrix[x,y,ind]
                    if spp == species_code:
                        basal_sum += basal_area_matrix[x,y,ind]
                grouped_basal_area_matrix[x,y,spp] = basal_sum

    return grouped_basal_area_matrix


def compute_biomass_by_plot_by_species(biomass_matrix, species_code_matrix, number_of_species):
    """
    Group biomass by plot and by species. Get rid of the -1 species (-1 is a flag for no tree/no species).
    The result contains only the plot by plot average biomass values for each species present on the plot.

    Parameters:  biomass_matrix          --  records the biomass of each tree in sim, size: sim grid by number of trees on each plot
                 species_code_matrix     --  records the specie of each tree in sim, size: sim grid by number of trees on each plot
                 number_of_species       --  the number of species in the simulation

    Returns:     biomass_matrix          --  returns a biomass (kg) summed up for every species on a plot
                                             size: nx,ny,spp_in_sim
    """
    nx, ny, nind = biomass_matrix.shape
    grouped_biomass_matrix = np.zeros((nx,ny,number_of_species))
    return compute_biomass_by_plot_by_species_numba(biomass_matrix, species_code_matrix, number_of_species, grouped_biomass_matrix)

@numba.jit(nopython=True)
def compute_biomass_by_plot_by_species_numba(biomass_matrix, species_code_matrix, number_of_species, grouped_biomass_matrix):
    nx, ny, nind = biomass_matrix.shape

    for x in range(nx):
        for y in range(ny):
            for spp in range(number_of_species):
                biomass_sum = 0.
                for ind in range(nind):
                    species_code = species_code_matrix[x,y,ind]
                    if spp == species_code:
                        biomass_sum += biomass_matrix[x,y,ind]
                grouped_biomass_matrix[x,y,spp] = biomass_sum

    return grouped_biomass_matrix


def compute_stems_by_plot_by_species(DBH_matrix, species_code_matrix, number_of_species):
    """
    Calculating stem density for stems >=?cm in DBH, for each species in sim, for each plot in sim grid.

    Parameters:  DBH_matrix              --  records DBH for every tree in sim, size: sim grid by number of trees on each plot
                 species_code_matrix     --  records the specie of each tree in sim, size: sim grid by number of trees on each plot
                 species_in_sim          --  the number of species in the simulation

    Returns:     DBH_distribution_matrix --  for each plot, count the number of stems >=8cm @ DBH for each species present
                                             size: nx, ny, spp_in_sim
    """
    nx,ny,ntrees = DBH_matrix.shape
    stems_matrix = np.zeros((nx,ny,number_of_species))
    THRESHOLD = 4.  #min DBH in cm used to count stem density
    return compute_stems_by_plot_by_species_numba(DBH_matrix, species_code_matrix, number_of_species, stems_matrix, THRESHOLD)

@numba.jit(nopython=True)
def compute_stems_by_plot_by_species_numba(DBH_matrix, species_code_matrix, number_of_species, stems_matrix, THRESHOLD):
    nx, ny, nind = DBH_matrix.shape

    for x in range(nx):
        for y in range(ny):
            for spp in range(number_of_species):
                stem_count = 0
                for ind in range(nind):
                    species_code = species_code_matrix[x,y,ind]
                    dbh = DBH_matrix[x,y,ind]
                    if spp == species_code:
                        if dbh > THRESHOLD:
                            stem_count += 1
                stems_matrix[x,y,spp] = stem_count

    return stems_matrix


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("driver_file", help="specify the driver file and path (if not in same directory)")
    parser.add_argument("output_file", help="specify the name of output HDF file for each model run")
    parser.add_argument("--no_numba", help="do not use numba (for debugging)", action="store_true")
    args = parser.parse_args()
    driver_file = args.driver_file
    output_filename = args.output_file
    if args.no_numba:
        use_numba_dummy = True
        force_rewrite = True
        print '!!! REMEMBER TO SWITCH TO "import numba_dummy as numba" AT THE TOP OF THIS FILE !!!'
    else:
        use_numba_dummy = False
        force_rewrite = False

    ######################
    #### Generate driver specific code that is specialized for fast operation with numba. 
    from create_specialized_driver_code import make_numba_driver_code
    make_numba_driver_code(driver_file=driver_file, output_file='specialized_driver_numba.py', 
                           use_numba_dummy=use_numba_dummy, force_rewrite=force_rewrite)
    from specialized_driver_numba import compute_species_factors_weather, compute_species_factors_soil, \
                                     compute_available_light_factors_by_species, \
                                     compute_individual_tree_values, \
                                     compute_actual_leaf_area, \
                                     number_of_species, species_code_to_name, species_code_to_CP, species_code_to_SEED, \
                                     species_code_to_age_mortality_function, species_code_to_inseeding_lag, name_to_species_code, species_code_to_stress_threshold
    ######################
    ### Load the driver file within the same scope as the the simulator code will run. This means that all definitions present in
    ### the driver will be available to the simulation. However!! the only one we use is 'driver' which should be a dictionary.
    exec(open(driver_file).read())
    # driver should now be available

    StartSimulation(driver_file, output_filename, driver)
