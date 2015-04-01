from math import * 
import numpy as np

######### helper function to fix numba bug ################
uID = 0
def numba_fix(fn):
    global uID
    fn.__name__ = fn.__name__ + '__%s' % uID
    uID += 1
    return fn
###########################################################


###########################################
## Begin generic function factory forms
###########################################
## the standard way of compute age mortality : from zelig2.3
def standard_age_mortality_probablity(AGEMAX):
    AGEMAX_ = float(AGEMAX)
    def standard_age_mortality_probablity_fn():
        return (1.0 - exp(-4.605 / AGEMAX_))
    return numba_fix(standard_age_mortality_probablity_fn)

## standard basal area calculation : from zelig1.0
def standard_basal_area(dbh):
    # area of a tree slice at breast height
    return pi*((0.5*(dbh/100.0))**2)

## standard way to calculate the optimal growth : from zelig1.0
def standard_optimal_growth_increment(HTMAX, DMAX, G):
    HTMAX_ = float(HTMAX); DMAX_ = float(DMAX); G_ = float(G)
    def standard_optimal_growth_increment_fn(dbh):
        '''
        Compute the optimal growth increment in cm.

        Parameters: dbh -- the tree dbh in cm

        Returns: the maximum amount this tree can grow (cm) in the next year
        '''
        # compute the growth constants B2 and B3
        B2 = 2.0 * ((HTMAX_ - 137.0) / DMAX_)
        B3 = (HTMAX_ - 137.0) / DMAX_**2
        #what's in ZELIG v1.0 code:
        gr = (137.0 + 0.25 * B2**2 / B3) * (0.5 * B2/B3)
        dm = G * dbh * (1.0 - (137.0 * dbh + B2 * dbh**2 - B3 * dbh**3) / gr) / (274.0 + 3.0 * B2 * dbh - 4.0 * B3 * dbh**2)
        return dm
    return numba_fix(standard_optimal_growth_increment_fn)

# standard way to compute tree height from dbh : from zelig1.0
def standard_tree_height(HTMAX, DMAX):
    HTMAX_ = float(HTMAX); DMAX_ = float(DMAX);
    def standard_tree_height_fn(dbh):
        '''
        Compute the tree height in meters for using species specific information from the driver.

        Parameters : dbh -- the tree dbh in cm

        Returns : height in meters
        '''
        # compute the growth constants B2 and B3
        B2 = 2.0 * ((HTMAX_ - 137.0) / DMAX_)
        B3 = (HTMAX_ - 137.0) / DMAX_**2
        if dbh > 0.0:
            return (137.0 + B2 * dbh - B3 * dbh**2)/100.0
        else:
            return 0.
    return numba_fix(standard_tree_height_fn)

# standard way to compute the degree day factor : from zelig1.0
def standard_degree_day_factor(DDMIN, DDMAX):
    DDMIN_ = float(DDMIN); DDMAX_ = float(DDMAX)
    def standard_degree_day_factor_fn(growing_degree_days):
        '''
        Compute the growing degree day factor. This factor will possibly limit
        growth due to the number of growing degree days, dependent on where
        the number of growing degree days is within the parabolic curve
        defined by the species-specific minimum and maximum growing
        degree day tolerances.

        Parameters: growing_degree_days -- annual sum of growing degree days

        Returns: degree_day_factor -- degree day factor limiting growth due to weather (0.0 to 1.0)
        '''
        degree_day_factor = 4.0*(growing_degree_days - DDMIN_) * (DDMAX_ - growing_degree_days) / (DDMAX_ - DDMIN_)**2
        # the factor can't be less than 0.0
        if degree_day_factor < 0.0: 
            degree_day_factor = 0.0
        # or greater than 1.0
        elif degree_day_factor > 1.0: 
            degree_day_factor = 1.0
        return degree_day_factor
    return numba_fix(standard_degree_day_factor_fn)

# one sided parabola of the standard way to compute the degree day factor : from zelig1.0
def half_parabola_degree_day_factor(DDMIN, DDMAX):
    DDMIN_ = float(DDMIN); DDMAX_ = float(DDMAX)
    MID_POINT_ = DDMAX_ - (DDMAX_ - DDMIN_)/2.
    def half_parabola_degree_day_factor_fn(growing_degree_days):
        '''
        Compute the growing degree day factor. This factor will possibly limit
        growth due to the number of growing degree days, dependent on where
        the number of growing degree days is within the parabolic curve
        defined by the species-specific minimum and maximum growing
        degree day tolerances.
        This version only penalizes for too cold. There is no too hot side.

        Parameters: growing_degree_days -- annual sum of growing degree days

        Returns: degree_day_factor -- degree day factor limiting growth due to weather (0.0 to 1.0)
        '''
        if growing_degree_days >= MID_POINT_:
            degree_day_factor = 1.0
        else:
            degree_day_factor = 4.0*(growing_degree_days - DDMIN_) * (DDMAX_ - growing_degree_days) / (DDMAX_ - DDMIN_)**2
        # the factor can't be less than 0.0
        if degree_day_factor < 0.0: 
            degree_day_factor = 0.0
        # or greater than 1.0
        elif degree_day_factor > 1.0: 
            degree_day_factor = 1.0
        return degree_day_factor
    return numba_fix(half_parabola_degree_day_factor_fn)

# standard way to compute soil moisture factor : from zelig1.0
def standard_soil_moisture_factor(MDRT):
    MDRT_ = float(MDRT)
    def standard_soil_moisture_factor_fn(drydays):
        '''
        Compute the species-specific tolerance of soil moisture factor, or otherwise
        known as the dry-day constraint factor.

        Parameters: drydays -- percent of dry days for the growing season

        Returns: soil_moisture_factor -- soil moisture factor range 0.0 to 1.0
        '''
        # if dry days = 0 then relative factor = 1.0 (not limited by drought)
        # if dry days < MDRT/10 then relative factor between 0.0 and 1.0 (some drought limit)
        # if dry days > MDRT/10 then relative factor = 0.0 (no growth due to drought)
        dt = MDRT_ / 10.0   #\10.0 to bring it to 0-to-1 value (so can have values 0.1-0.5, depending on drought
                           # tolerance class 1-5
        drt = min(dt, drydays)
        return ((dt - drt)/dt)**0.5
    return numba_fix(standard_soil_moisture_factor_fn)

# standard way to compute soil fertility factor : from zelig2.3
def standard_soil_fertility_factor(NUTRI):
    #NUTRI tolerance classes 1-3 in list
    S1_look_up_table = [1.03748,1.00892,1.01712] 
    S2_look_up_table = [-4.02952,-5.38804,-4.12162] 
    S3_look_up_table = [0.17588,0.12242,0.00898]
    S1 = S1_look_up_table[NUTRI-1]
    S2 = S2_look_up_table[NUTRI-1]
    S3 = S3_look_up_table[NUTRI-1]

    def standard_soil_fertility_factor_fn(sf):
        '''
        Computes nutrient response curves based on the tree's nutrient stress tolerance class
        (1=INTOL, 3=TOL) and soil fertility (i.e. limit on how many Mg/ha/yr that can be accrued in biomass)

        Parameters : sf -- soil fertility (0 to 1)

        Returns : fertf -- soil fertility factor (0 to 1); this is how well the tree responds to the input 
                           soil fertility
        ''' 
        #newer equation from ZELIG v2.3:
        fertf = S1 * (1.0 - exp(S2 * (sf - S3)))
        if fertf < 0.0:
            fertf = 0.0
        elif fertf > 1.0:
            fertf = 1.0
        return fertf
    return numba_fix(standard_soil_fertility_factor_fn)

# standard way to compute available light factor : from zelig1.0
def standard_available_light_factor(LIGHT):
    # the light tolerance constants
    # coefs for 1 to 5 integer of shade tolerance (1 is more shade tolerant, 5 is shade intolerant)
    C1_look_up_table = [1.01,1.04,1.11,1.24,1.49]  
    C2_look_up_table = [4.62,3.44,2.52,1.78,1.23]
    C3_look_up_table = [0.05,0.06,0.07,0.08,0.09]
    C1 = C1_look_up_table[LIGHT-1]
    C2 = C2_look_up_table[LIGHT-1]
    C3 = C3_look_up_table[LIGHT-1]
    def standard_available_light_factor_fn(al):
        avail_light_factor = C1 * (1.0 - exp(-1.*C2 * (al - C3) ) )
        # make sure we don't go above 1.0
        if avail_light_factor > 1.0: 
            avail_light_factor=1.0
        # make sure we don't go below 0.0
        elif avail_light_factor < 0.0: 
            avail_light_factor = 0.0  
        return avail_light_factor
    return numba_fix(standard_available_light_factor_fn)

########### End generic function factory forms ##########################

# standard way to calculate the light compensation point
def standard_light_compensation_point(LIGHT):
    # light compensation points (CP) for 5 shade-tolerance classes:
    # ??? DON'T UNDERSTAND HOW THESE ARE DERIVED, FOR NOW JUST USING VROM ZELIG V2.3 line 281
    CP_look_up_table = [0.15, 0.12, 0.09, 0.06, 0.03]
    return CP_look_up_table[LIGHT-1]

# standard way to calculate the inseeding lag time
def standard_inseeding_lag(LIGHT):
    #index by light tolerance class (1-5), so that shade-tolerant species have a longer lag time to sapling generation
    species_lag_look_up_table = [6,5,4,3,2]  #the lag is in reverse order because LIGHT=1 is for shade-tolerant, 
                                             #whereas LIGHt=5 is for shade-intolerant species  
    return species_lag_look_up_table[LIGHT-1]


############### species specific height equations #######################
def absi_tree_height(dbh):
    return -0.0049 * dbh**2 + 0.9546*dbh + 1.37

def lasi_tree_height(dbh):
    if dbh <= 36.0:
        return -0.0152*dbh**2 + 1.3806*dbh + 1.37
    else:
        return 6.0278*log(dbh) + 9.6025
    #return 11.738*dbh**0.3295

def betula_tree_height(dbh):
    if dbh <= 26.5:
        return 1.0389*dbh + 1.37
    else:
        return 1.3444*log(dbh) + 24.501

def piob_height_piecwise_fn(dbh):
    if dbh <= 25.0:
        return 0.0401*dbh**2 - 0.0516*dbh + 1.37
    else:
        return 21.928*log(dbh) - 45.405

def pisi_height_piecwise_fn(dbh):
    #if dbh <= 51.6:
    #    return -0.0076*dbh**2 + 0.9224*dbh + 1.37
    #else:
    #    return 14.596*log(dbh) - 28.704
    if dbh <=38.5:
        return -0.0073*dbh**2 + 0.913*dbh + 1.37
    else:
        return 9.3544*log(dbh) - 8.4582

def pisy_height_piecwise_fn(dbh):
#    if dbh <= 17.6:
#        return 11.584*dbh**0.3368
#    else:
#        return 8.0044*log(dbh) + 7.4694
    if dbh <= 32.0:
        return -0.0105*dbh**2 + 1.1644*dbh + 1.37
    else:
        return 12.739*log(dbh)-16.297

def populus_tree_height(dbh):
    if dbh <= 34.0:
        return 0.0013*dbh**2 + 0.736*dbh + 1.37
    else:
        return 13.358*log(dbh) - 19.19
############### end species specific height equations ###################


################ species specific leaf area equations ###################
def absi_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=116.
    if dbh <= 33.0:
        foliar_biomass = 0.015*dbh**2.1934
    else:
        foliar_biomass = 53.975*log(dbh) - 156.59
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA

def lasi_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=229.
    if dbh <= 34.0:
        foliar_biomass = 0.0218*dbh**2.0014
    else:
        foliar_biomass = 69.793*log(dbh) - 220.71
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA

def betula_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=199.
    if dbh <= 22.4:
        foliar_biomass = 0.0492*dbh**1.5835
    else:
        foliar_biomass = 6.5771*log(dbh) - 13.672
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA

def piob_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=129.
    if dbh <= 27.3:
        foliar_biomass = 0.0497*dbh**1.844
    else:
        foliar_biomass = 13.789*log(dbh) - 23.48
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA

def pisi_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=152.
    if dbh <= 33.6:
        foliar_biomass = 0.0075*dbh**2 + 0.2328*dbh - 0.2592
    else:
        foliar_biomass = 19.537*log(dbh) - 52.641
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA

def pisy_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=134.
    if dbh <= 40.0:
        foliar_biomass = 0.0298*dbh**1.7463
    else:
        foliar_biomass = 15.907*log(dbh) - 39.909
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA

def populus_leaf_area_fn(dbh):
    SPECIFIC_LEAF_AREA=159.   #for P.tremuloides SLA=135 in lowN and 146 in highN conditions (Curtis et al., 2000) 
    if dbh <= 28.3:
        foliar_biomass = 0.0281*dbh**1.6509
    else:
        foliar_biomass = 7.6636*log(dbh) - 18.632
    # kg * cm^2/g => m^2
    MASS_RATIO_TO_AREA = 10.
    return foliar_biomass * SPECIFIC_LEAF_AREA / MASS_RATIO_TO_AREA
################ end species specific leaf area equations ###############


########### species specific biovolume functions ############# 
def absi_biovolume(dbh):
    return 0.0001 * dbh**2.5371

def lasi_biovolume(dbh):
    if dbh <= 20.0:
        return 0.0004 * dbh**2.3061
    else:
        return 0.0013 * dbh**2 - 0.0074 * dbh
    #return 0.0013 * dbh**2 - 0.0074 * dbh

def bepe_biovolume(dbh):
    return 0.0002 * dbh**2.5213

def bepl_biovolume(dbh):
    return 0.0002 * dbh**2.5213

def bepu_biovolume(dbh):
    return 0.0002 * dbh**2.5213

def piob_biovolume(dbh):
    return 0.00006 * dbh**2.8291

def pisi_biovolume(dbh):
    return 0.0001 * dbh**2.514

def pisy_biovolume(dbh):
    return 0.0003 * dbh**2.4137

def posu_biovolume(dbh):
    return 0.0001 * dbh**2.5877

def potr_biovolume(dbh):
    return 0.0001 * dbh**2.5877
########### end species specific biovolume functions ##########


########### species specific biomass functions ################
def absi_total_biomass(dbh):
    #return 0.0001*dbh**2.3543
    return 0.00002*dbh**3 - 0.0003*dbh**2 + 0.0039*dbh

def lasi_total_biomass(dbh):
    return 0.0002*dbh**2.5568 #0.0045*dbh**1.6441

def betula_total_biomass(dbh):
    return 0.0002*dbh**2.3793

def piob_total_biomass(dbh):
    return 0.00006*dbh**2.6887

def pisi_total_biomass(dbh):
    #return 0.0006*dbh**2 - 0.0058*dbh + 0.0202
    if dbh <= 43.0:
        pisi_total_biomass = 0.00008 * dbh**2.5387
    else:
        pisi_total_biomass = 2.8067*log(dbh)-9.4117
    return pisi_total_biomass

def pisy_total_biomass(dbh):
    return 0.0001*dbh**2.3922

def populus_total_biomass(dbh):
    return 0.0001*dbh**2.4599
########### end species specific biomass functions #############

########### optimal increment function #################
def bragg_optimal_growth_increment(A, B, C, multiplier):  #factory
    A_ = float(A); B_ = float(B); C_ = float(C); multiplier_ = float(multiplier)
    def bragg_optimal_growth_increment_innerfn(dbh):  #fn
        """
        Compute the optimal growth increment in cm.

        Parameters: dbh -- the tree dbh in cm

        Returns: the maximum amount this tree can grow (cm) in the next year
        """
        return multiplier*(A_*(dbh**B_)*(C_**dbh))*dbh
    return numba_fix(bragg_optimal_growth_increment_innerfn)
########### end optimal increment functions ############

####################################################
#             BEGIN driver dictionary              #
####################################################

MAX_TREES_PER_PLOT = 50 #24 #47
driver = {
# enter some text here that describes any notes about this simulation run; why, where, how, ...
"run_description":
"""
This simulation contains all 10 boreal tree species.
""",


"TITLE":"ZELIG-BORK version 4.0 (under testing)",
"sim_start_year":0,
"sim_stop_year":20,
"LIGHT_MODE":"3D",
#DEBUG True saves the factors to see where things may have gone wrong
"DEBUG":True,

# Set LOAD_INTIIAL_CONDITIONS to True to set the "state" of the simulation when it starts
"LOAD_INTIIAL_CONDITIONS":False,
"INITIAL_CONDITIONS":{"DEM_matrix": np.zeros((181,12,MAX_TREES_PER_PLOT)) + 5.8 + 0.25**np.random.random(((181,12,MAX_TREES_PER_PLOT))),  # set all of the dbh values to 1.6 cm; size nx,ny,MAX_TREES_PER_PLOT
                      "CrownBase_matrix": np.zeros((181,12,MAX_TREES_PER_PLOT), dtype=np.int),      # all of the trees have the crown base at the ground
                      "SpeciesCode_matrix": np.zeros((181,12,MAX_TREES_PER_PLOT), dtype=np.int)+1},   # set all of the species code values to 0 (the only species in the sim right now)

# Set ALLOW_PLANTING to True to allow yearly planting when space is available.
# Only set this to False when using initial conditions, otherwise the simulation will only contain bare ground.
"ALLOW_REGENERATION":True,

"CLIMATE_RANDOM_doc":"if true, then generate random climate; if false, then use climate record specified below",
"CLIMATE_RANDOM":False,

"TIMBER_HARVEST_doc":"if true, trees of specified spp and size are removed within one year of sim; if false, harvest not activated",
"TIMBER_HARVEST":False,

"INSECT_OUTBREAK_doc":"if true, a combination of warm spring and stressed trees result in greater mortality than SMORT & AMORT; if false, not activated",
"INSECT_OUTBREAK":False,

"WILDFIRE_doc":"if true, fire cycle and fuel load trigger fire, like in Bonan 1988; if false, not activated",
"WILDFIRE":False,

"LOCALE":"Central Siberian boreal forest",
"PHIB":0.450,
"PHID":0.550,
"FC":45.00,
"WP":22.50,
"DDBASE":5,
"MAX_TREES_PER_PLOT":MAX_TREES_PER_PLOT,
"MAX_TREE_HEIGHT":50,         

"XT_VT_XR_VR_doc":"QAed temps, 2/3stdevs; XT=avg monthly temp in degrees C; VT=std for monthly temps; XR=average monthly sums of precip in cm; VR=std for monthly precip; maxT=max daily temp obs over 50+ yrs of WMO data record",
"XT": np.array([-20.53, -18.9, -10.13, 0.30, 8.55, 15.51, 18.66, 15.18, 7.92, -0.12, -10.60, -18.04]),
#"VT": [  0.,     0.,    0.,   0.,   0.,    0.,    0.,    0.,   0.,    0.,     0.,     0., ],
"VT": [10.17,   9.08,   7.36, 5.2,  4.97,  4.17,  2.98,  3.36, 3.82,  5.2,    8.6,   10.02],
#"maxT": [5.7,   8.1,    13.5, 27.8, 33.4,  34.4,  34.9,  34.4, 32.1,  22.9,   13.6,   6.1],  # max temperature of daily max by month
"maxT": [  2.76,  2.125,  4.9,   14.68, 23.67, 26.65, 26.9, 24.8, 18.6,  15.5,   5.87,  2.13],         # max temperature of daily means by month
"minT": [-48.6, -42.67, -36.36, -25.1,  -5.97,  2.7,   9.58, 5.2, -2.5, -22.0, -37.0, -48.2],          # min temperature of daily means by month
"XR": [2.05, 1.49, 1.47, 2.21,  3.86,  4.83,  6.09,  6.20,  4.45, 3.66, 3.49, 2.49],
#"VR": [0.,   0.,   0.,   0.,    0.,    0.,    0.,    0.,    0.,   0.,   0.,   0., ],
"VR": [0.94, 0.81, 1.09, 1.16,  1.57,  2.35,  3.04,  3.14,  2.05, 1.78, 1.41, 1.97],

"AvgRadiation":[16.75, 51.84, 111.74, 179.89, 207.65, 236.91, 224.50, 264.02, 102.64, 47.59, 22.16, 10.00],
"StdRadiation":[1.49,  4.60,  8.94,  15.3,  19.33,  22.21,  23.34,  21.77,  18.08,  6.40,  2.88,  1.01],

"FILEPATH_doc":"filepaths below point ZELIG-BORK to spatially explicit driver files, which must be kept in same folder as the model code",
"DEM_file_path":"elevation10mFlat180mamsl.txt",
"elevation_lapse_rate_adjustment_matrix_filepath":"elev180_adj_factor.txt",
"monthly_radiation_files_path":"monthly_radiation",
"Radiation_fraction_file_path":"rad_fraction.txt", 
"Site_index_file_path":"siteindex.txt",

# function gets called once per year: inputs=year, outputs=boolean (True means write this years data to HDF, False no write)
"log_this_year_function": lambda year: (year % 1) == 0,

"SPECIES_doc":"below are silvicultural and tolerance information for each species in sim; set ENABLED=False to eliminate spp from sim",
"species":{## the format is as follows
           #"SPECIES NAME":{
           #        "ENABLED":True or False,
           #
           #        "SEED": float value (make it a ratio to competetitors),
           #
           #        "DMAX": float estimate of maximum tree diameter (cm),
           #
           #        "BIOVOLUME_EQUATION": function that accepts float dbh (cm) and returns float biovolume (m^3),
           #
           #        "BIOMASS_EQUATION": function that accepts float dbh (cm) and returns float biomass (Mg),
           #
           #        "LEAF_AREA_EQUATION": function that accepts float dbh (cm) and returns float leaf area (m^2),
           #
           #        "AGE_MORTALITY_EQUATION": function with no inputs but outputs a float between 0.0 and 1.0,
           #
           #        "BASAL_AREA_EQUATION": function that accepts float dbh (cm) and returns float basal area (m^2),
           #
           #        "OPTIMAL_GROWTH_INCREMENT_EQUATION": function that accepts float dbh (cm) and returns float optimal increment (cm),
           #
           #        "TREE_HEIGHT_EQUATION": function that accepts float dbh (cm) and returns height (m),
           #
           #        "SOIL_FERTILITY_FACTOR_EQUATION": function that accepts float soil relative soil fertility (0 to 1) and returns float species specific response (0 to 1),
           #
           #        "SOIL_MOISTURE_FACTOR_EQUATION": function that accepts float soil relative dry days (0 to 1) and returns float species specific response (0 to 1),
           #
           #        "DEGREE_DAY_FACTOR_EQUATION": function that accepts float growing degree days and returns float species specific response (0 to 1),
           #
           #        "AVAILABLE_LIGHT_FACTOR_EQUATION": function that accepts float relative available light (0 to 1) and returns float species specific response (0 to 1),
           #
           #        "LIGHT_COMPENSATION_POINT":  float species specific compensation point (0 to 1),
           #
           #        "INSEEDING_LAG": integer species specific inseeding lag in years
           #       },
           "ABSI":{
                   "ENABLED":True,
                   "SEED":0.31,
                   #"OPT_INC_MULTIPLIER": 4.0,    #DECENT ON THE FIRST RUN, CAN TWEAK A BIT TO BETTER REPRESENT LARGER/OLDER TREES
                   #"INITIAL_DBH": 3.8,
                   #"MAX_TREES_PER_PLOT": 73,
                   #"START&STOP_YEARS": 20, 190,
                   #"MAX_BIOVOLUME_INCREMENT": 4.0,
                   "DMAX":80.,
                   "BIOVOLUME_EQUATION": absi_biovolume, #lambda dbh: 0.0001 * dbh**2.5371,
                   "BIOMASS_EQUATION": absi_total_biomass, #lambda dbh: 7.7962 * (dbh**0.906),
                   "LEAF_AREA_EQUATION": absi_leaf_area_fn, #lambda dbh: 0.160694 * (dbh**2.129),
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=300.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=4000., DMAX=80., G=83.),   #JABOWA
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.24946, B=-0.327, C=0.91, multiplier = 1.0),#4.0),        #BRAGG
                   "TREE_HEIGHT_EQUATION": absi_tree_height, #lambda dbh: -0.0049 * dbh**2 + 0.9546*dbh + 1.37, #        standard_tree_height(HTMAX=4000., DMAX=80.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=1),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=1.),
                   #"DEGREE_DAY_FACTOR_EQUATION": standard_degree_day_factor(DDMIN=510., DDMAX=1530.),#DDMAX=1450.),
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=510., DDMAX=1450.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=1),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=1),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=1),
                   #Yield Table values(p.260,593):
#                   "EXPECTED_AGE_DBH": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.],                        
#                                        [3.8,8.1,12.7,17.,20.7,23.9,26.4,28.4,29.9,32.,33.2,33.9,34.3]),
#                   "EXPECTED_HEIGHT": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.],
#                                       [5.,9.,12.8,16.2,19.,21.2,23.,24.4,25.4,26.9,27.7,28.1,28.4]),
#                   "EXPECTED_STEMS": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.],
#                                      [7353.,3213.,1764.,1131.,812.,635.,528.,461.,416.,364.,339.,325.,318.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.],
#                                               [8.3,16.4,22.2,25.6,27.4,28.4,28.9,29.1,29.3,29.4,29.4,29.4,29.4]),
#                   "EXPECTED_AGE_BIOVOLUME": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.],
#                                              [33.,84.,143.,196.,240.,273.,297.,314.,326.,340.,346.,349.,351.]),
#                   "EXPECTED_AGE_BIOMASS": ([20.,40.,60.,80.,100.,120.,140.,160.,180.], 
#                                            [25.5, 82.1,124.6,149.4,164.2,173.7,182.,190.6,200.6])
                  },

           "LASI":{
                   "ENABLED":True,
                   "SEED":0.31,               #PARAMETERIZED FOR THIS SPECIES, NEED TO TWEAK A BIT TO BETTER REPRESENT MIDDLE-SIZED TREES
                   #"OPT_INC_MULTIPLIER": 5.0,
                   #"INITIAL_DBH": 11.3,
                   #"MAX_TREES_PER_PLOT": 22,
                   #"START&STOP_YEARS": 40, 260,
                   #"MAX_BIOVOLUME_INCREMENT": 2.3,
                   "DMAX":100.,
                   "BIOVOLUME_EQUATION": lasi_biovolume, #lambda dbh: 0.0002 * dbh**2.3479,
                   "BIOMASS_EQUATION": lasi_total_biomass, #lambda dbh: 86.156 * (dbh**0.5977),
                   #"LEAF_AREA_EQUATION": lambda dbh: 0.160694 * (dbh**2.129),
                   "LEAF_AREA_EQUATION": lasi_leaf_area_fn,
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=400.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=4500., DMAX=100., G=100.),  #JABOWA
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.06768, B=0.063435, C=0.9241, multiplier = 1.0),#1.4),#5.0),     #BRAGG
                   "TREE_HEIGHT_EQUATION": lasi_tree_height, #lambda dbh: 5.2436 * dbh**0.487, #                     standard_tree_height(HTMAX=4500., DMAX=100.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=3),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=5),
                   #"DEGREE_DAY_FACTOR_EQUATION":      standard_degree_day_factor(DDMIN=300., DDMAX=1720.), #standard_degree_day_factor(DDMIN=300., DDMAX=1500.),
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=300., DDMAX=1500.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=5),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=5),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=5),
                   #Yield Table values(p.231,545):
#                   "EXPECTED_AGE_DBH": ([40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180,200.,220.,240.,250.],                        
#                                        [11.3,15.,18.6,22.,25.2,28.1,30.8,33.3,35.5,37.4,39.2,40.8,42.2,44.6,46.4,47.8,49.,49.4]),
#                   "EXPECTED_STEMS": ([40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180,200.,220.,240.,250.],
#                                      [2170.,1318.,883.,638.,489.,393.,328.,282.,248.,222.,202.,187.,175.,157.,145.,136.,130.,127.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180,200.,220.,240.,250.],
#                                               [21.6,23.1,23.9,24.2,24.3,24.4,24.4,24.4,24.5,24.5,24.5,24.5,24.5,24.5,24.5,24.5,24.5,24.5]),
#                   "EXPECTED_AGE_BIOVOLUME": ([40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180,200.,220.,240.,250.],
#                                              [201.,236.,264.,287.,304.,317.,327.,335.,341.,346.,349.,352.,354.,357.,358.,357.,355.,353.]),
#                   "EXPECTED_AGE_BIOMASS": ([20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.], 
#                                            [122.,234.1,326.5,398.8,454.,495.5,526.1,548.6,565.,576.6,585.1])
                  },

           "BEPE":{
                   "ENABLED":True,
                   "SEED":0.1,
                   #"OPT_INC_MULTIPLIER": 2.0,             #DECENT ON THE FIRST TRY, NEED TO TWEAK A BIT TO BETTER REPRESENT OLDER/LARGER TREES
                   #"INITIAL_DBH": 6.5,
                   #"MAX_TREES_PER_PLOT": 24,
                   #"START&STOP_YEARS": 15, 130,
                   #"MAX_BIOVOLUME_INCREMENT": 3.99,
                   "DMAX":60.,
                   "BIOVOLUME_EQUATION": bepe_biovolume, #lambda dbh: 0.0002 * dbh**2.5213,
                   "BIOMASS_EQUATION": betula_total_biomass, #lambda dbh: 7.0162 * dbh,
                   #"LEAF_AREA_EQUATION": lambda dbh: 0.160694 * (dbh**2.129),
                   "LEAF_AREA_EQUATION": betula_leaf_area_fn,
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=120.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=3000., DMAX=60., G=220.),   #JABOWA
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.12369, B=0.51527, C=0.8187, multiplier = 1.0),#1.2),#2.0)      #BRAGG
                   "TREE_HEIGHT_EQUATION": betula_tree_height, #lambda dbh: 1.0401*dbh + 1.37, #                          standard_tree_height(HTMAX=3000., DMAX=60.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=2),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=4),
                   #"DEGREE_DAY_FACTOR_EQUATION":      standard_degree_day_factor(DDMIN=410., DDMAX=2300.),
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=410., DDMAX=2300.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=5),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=5),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=5),
                   #Yield Table values(p.323,666):
#                   "EXPECTED_AGE_DBH": ([15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.,100.,110.,120.],                        
#                                        [6.5,9.4,12.1,14.5,16.5,18.3,19.8,21.1,23.,24.3,25.2,25.7,26.1,26.4,26.5]),
#                   "EXPECTED_HEIGHT": ([15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.,100.,110.,120.],
#                                       [8.1,11.1,13.9,16.4,18.5,20.3,21.9,23.2,25.2,26.6,27.5,28.2,28.6,28.9,29.2]),
#                   "EXPECTED_STEMS": ([15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.,100.,110.,120.],
#                                      [2416.,1608.,1196.,955.,801.,697.,623.,569.,498.,456.,430.,413.,400.,390.,383.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.,100.,110.,120.],
#                                               [8.1,11.2,13.7,15.7,17.2,18.4,19.2,19.9,20.7,21.1,21.3,21.5,21.4,21.3,21.2]),
#                   "EXPECTED_AGE_BIOVOLUME": ([15.,20.,25.,30.,35.,40.,45.,50.,60.,70.,80.,90.,100.,110.,120.],
#                                              [42.,68.,95.,119.,140.,157.,172.,184.,210.,212.,219.,223.,225.,227.,228.]),
#                   "EXPECTED_AGE_BIOMASS": ([15.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.], 
#                                            [40.6,63.,103.3,133.1,152.6,165.,172.5,176.9,179.5,181.,181.6,181.9])
                  },

           "PIOB":{
                   "ENABLED":True,
                   "SEED":0.31,
                   #"OPT_INC_MULTIPLIER": 4.0,             #DECENT ON THE FIRST TRY, NEED TO TWEAK A BIT TO BETTER REPRESENT OLDER/LARGER TREES & DECREASE STEM#
                   #"INITIAL_DBH": 8.8,
                   #"MAX_TREES_PER_PLOT": 9,
                   #"START&STOP_YEARS": 20, 210,
                   #"MAX_BIOVOLUME_INCREMENT": 3.92,
                   "DMAX":50.,
                   "BIOVOLUME_EQUATION": piob_biovolume, #lambda dbh: 0.00006 * dbh**2.8291,
                   "BIOMASS_EQUATION": piob_total_biomass, #lambda dbh: 0.6888 * dbh,
                   #"LEAF_AREA_EQUATION": lambda dbh: 0.160694 * (dbh**2.129),
                   "LEAF_AREA_EQUATION": piob_leaf_area_fn,
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=500.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=4000., DMAX=50., G=83.),
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.11245, B=0.06395, C=0.88756, multiplier = 1.0),#4.0),
                   "TREE_HEIGHT_EQUATION": piob_height_piecwise_fn, #            standard_tree_height(HTMAX=4000., DMAX=50.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=2),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=1),
                   #"DEGREE_DAY_FACTOR_EQUATION":      standard_degree_day_factor(DDMIN=320., DDMAX=1500.),
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=320., DDMAX=1500.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=1),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=1),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=1),
                   #Yield Table values(p.204,533):
#                   "EXPECTED_AGE_DBH": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.],                        
#                                        [8.8,12.8,16.4,19.5,22.1,24.4,26.4,28.2,29.6,30.9,32.,32.9,33.7,34.4,35.,35.9,36.6]),
#                   "EXPECTED_HEIGHT": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.],
#                                       [3.1,7.2,11.8,16.3,20.1,23.3,25.8,27.7,29.1,30.2,31.,31.5,32.,32.3,32.5,32.8,32.9]),
#                   "EXPECTED_STEMS": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.],
#                                      [677.,819.,788.,702.,612.,535.,473.,425.,388.,359.,336.,318.,303.,291.,282.,267.,258.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.],
#                                               [4.2,10.6,16.6,20.9,23.5,25.1,26.,26.5,26.7,26.9,27.,27.,27.,27.1,27.1,27.1,27.1]),
#                   "EXPECTED_AGE_BIOVOLUME": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.],
#                                              [18.,61.,121.,182.,234.,275.,304.,325.,339.,349.,356.,360.,363.,365.,366.,367.,368.]),
#                   "EXPECTED_AGE_BIOMASS": ([20.,40.,60.,80.,100.,120.,140.,160.,180.,200.], 
#                                            [14.1,85.9,158.3,201.4,222.6,231.7,235.8,237.9,239.2,240.4])
                  },

           "PISI":{
                   "ENABLED":True,
                   "SEED":0.56,
                   #"OPT_INC_MULTIPLIER": 4.0,             #DECENT ON THE FIRST TRY, NEED TO TWEAK A BIT TO BETTER REPRESENT OLDER/LARGER TREES
                   #"INITIAL_DBH": 9.8,
                   #"MAX_TREES_PER_PLOT": 27,
                   #"START&STOP_YEARS": 30, 310,
                   #"MAX_BIOVOLUME_INCREMENT": 6.49,
                   "DMAX":120.,
                   "BIOVOLUME_EQUATION": pisi_biovolume, #lambda dbh: 0.0001 * dbh**2.514,
                   "BIOMASS_EQUATION": pisi_total_biomass, #lambda dbh: 8.0933 * (dbh**1.0258),
                   #"LEAF_AREA_EQUATION": lambda dbh: 0.160694 * (dbh**2.129),
                   "LEAF_AREA_EQUATION": pisi_leaf_area_fn,
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=400.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=4500., DMAX=120., G=88.),
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.174, B=-0.4777, C=0.96367, multiplier = 1.0),#for young:4.0),
                   "TREE_HEIGHT_EQUATION": pisi_height_piecwise_fn, #                          standard_tree_height(HTMAX=4500., DMAX=120.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=2),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=3),
                   #"DEGREE_DAY_FACTOR_EQUATION":      standard_degree_day_factor(DDMIN=490., DDMAX=1660.), #DDMAX=1450.)
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=490., DDMAX=1450.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=2),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=2),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=2),
                   #Yield Table values(p.272,607):
#                   "EXPECTED_AGE_DBH": ([30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],                        
#                                        [9.8,14.,18.2,22.2,26.1,29.8,33.2,36.4,39.4,42.1,44.6,46.9,49.,50.9,54.2,56.9,59.2,61.,62.5,63.8,64.7]),
#                   "EXPECTED_HEIGHT": ([30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],
#                                       [9.5,12.8,15.8,18.3,20.5,22.2,23.7,24.9,25.9,26.7,27.4,27.9,28.3,28.7,29.2,29.5,29.7,29.9,30,30,30.1]),
#                   "EXPECTED_STEMS": ([30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],
#                                      [2681.,1912.,1417.,1087.,860.,699.,583.,497.,432.,382.,342.,311.,286.,265.,234.,212.,197.,185.,176.,170.,164.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],
#                                               [20.1,29.3,36.7,42.1,46.,48.7,50.5,51.7,52.5,53.1,53.4,53.7,53.8,53.9,54.,54.1,54.1,54.1,54.1,54.1,54.1]),
#                   "EXPECTED_AGE_BIOVOLUME": ([30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],
#                                              [105.,193.,287.,375.,454.,519.,573.,615.,649.,674.,694.,709.,720.,729.,740.,747.,750.,752.,753.,754.,754.]),
#                   "EXPECTED_AGE_BIOMASS": ([30.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.], 
#                                            [70.7,126.1,237.2,320.7,372.9,400.8,413.9,417.7,416.1,411.7,405.5,398.5,391.0,383.2,375.4])
                  },

           "PISY":{
                   "ENABLED":True,
                   "SEED": 0.31,
                   #"OPT_INC_MULTIPLIER": 4.0,             #DECENT ON THE FIRST TRY, NEED TO TWEAK A BIT TO BETTER REPRESENT MIDDLE-SIZED TREES
                   #"INITIAL_DBH": 4.8,
                   #"MAX_TREES_PER_PLOT": 135,
                   #"START&STOP_YEARS": 20, 190,
                   #"MAX_BIOVOLUME_INCREMENT": 9.07,  #8.5 p.728 SI1 fully-stocked
                   "DMAX":80.,
                   "BIOVOLUME_EQUATION": pisy_biovolume, #lambda dbh: 0.0003 * dbh**2.4137,
                   "BIOMASS_EQUATION": pisy_total_biomass, #lambda dbh: 35.226 * (dbh**0.619),
                   #"LEAF_AREA_EQUATION": lambda dbh: 0.160694 * (dbh**2.129),
                   "LEAF_AREA_EQUATION": pisy_leaf_area_fn,
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=300.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=3800., DMAX=80., G=140.),
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.2752, B=-0.52857, C=0.957, multiplier = 1.0),#2.5),#(A=0.4579, B=-0.532, C=0.961975), *2.2
                   "TREE_HEIGHT_EQUATION": pisy_height_piecwise_fn, #                          standard_tree_height(HTMAX=3800., DMAX=80.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=3),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=5),
                   #"DEGREE_DAY_FACTOR_EQUATION":      standard_degree_day_factor(DDMIN=450., DDMAX=2350.),
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=450., DDMAX=2350.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=4),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=4),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=4),
                   #Yield Table values(p.136,448):
#                   "EXPECTED_AGE_DBH": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.],                        
#                                        [4.8,9.5,14.6,19.7,24.7,29.3,33.4,37.1,40.3,43.1,45.5,47.6,49.3,50.8,52.1,53.2,54.1]),
#                   "EXPECTED_HEIGHT": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.],
#                                       [6.2,10.5,14.7,18.5,21.9,24.8,27.3,29.5,31.3,32.8,34,35.1,36,36.7,37.3,37.7,38.1]),
#                   "EXPECTED_STEMS": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.],
#                                      [13492.,4917.,2381.,1380.,907.,654.,505.,411.,348.,305.,273.,250.,232.,219.,208.,200.,193.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.],
#                                               [24.7,34.6,39.7,42.3,43.4,44.,44.2,44.3,44.4,44.4,44.4,44.4,44.4,44.4,44.4,44.4,44.4]),
#                   "EXPECTED_AGE_BIOVOLUME": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.],
#                                              [167.,270.,363.,442.,508.,562.,605.,639.,666.,687.,704.,717.,728.,736.,742.,747.,750.]),
#                   "EXPECTED_AGE_BIOMASS": ([20.,40.,60.,80.,100.,120.,140.,160.,180.], 
#                                            [89.4,191.6,267.8,319.2,352.4,373.5,386.9,395.4,400.9])
                  },

           "POTR":{
                   "ENABLED":True,
                   "SEED":0.1,
                   #"OPT_INC_MULTIPLIER": 4.0,             #DECENT ON THE FIRST TRY, NEED TO TWEAK A BIT TO BETTER REPRESENT OLDER/LARGER TREES
                   #"INITIAL_DBH": 5.0,
                   #"MAX_TREES_PER_PLOT": 78,
                   #"START&STOP_YEARS": 10, 120,
                   #"MAX_BIOVOLUME_INCREMENT": 8.0,
                   "DMAX":100.,
                   "BIOVOLUME_EQUATION": potr_biovolume, #lambda dbh: 0.0001 * dbh**2.5877,
                   "BIOMASS_EQUATION": populus_total_biomass, #lambda dbh: 8.1523 * dbh,
                   #"LEAF_AREA_EQUATION": lambda dbh: 0.160694 * (dbh**2.129),
                   "LEAF_AREA_EQUATION": populus_leaf_area_fn,
                   "AGE_MORTALITY_EQUATION": standard_age_mortality_probablity(AGEMAX=100.),
                   "BASAL_AREA_EQUATION": standard_basal_area,
                   #"OPTIMAL_GROWTH_INCREMENT_EQUATION": standard_optimal_growth_increment(HTMAX=3500., DMAX=100., G=300.),
                   "OPTIMAL_GROWTH_INCREMENT_EQUATION": bragg_optimal_growth_increment(A=0.4113, B=-0.3054, C=0.91328, multiplier = 1.0),#4.0),
                   "TREE_HEIGHT_EQUATION": populus_tree_height, #lambda dbh: 0.0013*dbh**2 + 0.736*dbh + 1.37, #   standard_tree_height(HTMAX=3500., DMAX=100.),
                   "SOIL_FERTILITY_FACTOR_EQUATION": standard_soil_fertility_factor(NUTRI=2),
                   "SOIL_MOISTURE_FACTOR_EQUATION": standard_soil_moisture_factor(MDRT=4),
                   #"DEGREE_DAY_FACTOR_EQUATION":      standard_degree_day_factor(DDMIN=400., DDMAX=3000.),
                   "DEGREE_DAY_FACTOR_EQUATION": half_parabola_degree_day_factor(DDMIN=400., DDMAX=3000.),
                   "AVAILABLE_LIGHT_FACTOR_EQUATION": standard_available_light_factor(LIGHT=4),
                   "LIGHT_COMPENSATION_POINT":      standard_light_compensation_point(LIGHT=4),
                   "INSEEDING_LAG":                            standard_inseeding_lag(LIGHT=4),
                   #Yield Table values(p.339,687):
#                   "EXPECTED_AGE_DBH": ([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.],                        
#                                        [5.,13.,20.3,26.1,30.4,33.5,35.7,37.2,38.3,39.,39.5]),
#                   "EXPECTED_HEIGHT": ([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.],
#                                       [6.,12.,17.,21.1,24.3,26.9,28.9,30.4,31.6,32.6,33.3]),
#                   "EXPECTED_STEMS": ([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.],
#                                      [7847.,1709.,840.,564.,443.,380.,345.,323.,309.,301.,295.]),
#                   "EXPECTED_AGE_BASAL_AREA": ([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.],
#                                               [15.4,22.6,27.1,30.1,32.1,33.5,34.5,35.1,35.6,35.9,36.1]),
#                   "EXPECTED_AGE_BIOVOLUME": ([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.],
#                                               [69.,157.,240.,311.,370.,418.,456.,487.,511.,530.,545.]),
#                   "EXPECTED_AGE_BIOMASS": ([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.], 
#                                            [54.8,113.4,163.9,205.5,239.2,265.6,286.5,302.5,315.,324.6,332.])
                  }, 



          },
#Totals from Yield Table (p.276) PISI/ABSI/LASI/BEPE: 
#"EXPECTED_AGE_DBH":([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],                        
#                    [3.9,6.2,8.6,10.9,13.1,15.3,17.4,19.4,21.5,23.,24.7,26.3,27.8,29.2,30.5,32.9,35.,36.9,38.5,39.9,41.2,42.2]),
#"EXPECTED_HEIGHT": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220,240.,260.,280.,300.],
#                    [2.1,4.6,7.3,10.1,12.7,15.,17.,18.7,20.1,21.2,22.1,22.9,23.5,24.,24.4,24.9,25.3,25.5,25.6,25.7,25.7,25.8]),
#"EXPECTED_AGE_BASAL_AREA": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.],
#                            [4.6,10.3,16.6,22.4,27.5,31.6,34.8,37.3,39.2,40.6,41.6,42.3,42.9,43.3,43.6,44.,44.2,44.3,44.4,44.4,44.4,44.4]),
#"EXPECTED_AGE_BIOVOLUME": ([20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,180.,200.,220.,240.,260.,280.,300.], 
#                           [17.,49.,93.,144.,198.,249.,296.,338.,374.,404.,430.,451.,468.,483.,494.,511.,522.,528.,533.,536.,537.,539.]),
#"EXPECTED_AGE_BIOMASS": ([15.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.], 
#                         [40.6,63.,103.3,133.1,152.6,165.,172.5,176.9,179.5,181.,181.6,181.9])
}
