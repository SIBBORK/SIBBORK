import numpy as np   # numpy for numerical code (arrays, etc.)
from read_in_ascii import read_in_ascii #from filename import function
from make_ERDASimg import generate_ERDASimg_grid

#Degree day base for boreal forest photosynthesis 5.0 degrees C
ddbase = 5.0

#######################################################################
def GrowingDegreeDays_calc(ddbase, monthly_temperature_avgs_lst, monthly_temperature_stds_lst, 
                           monthly_temperature_mins_lst, monthly_temperature_maxs_lst, lapse_rate_adj_mat):
    """
    Simulate monthly temperature based off of driver monthly means and standard deviations;
    add elev/lapse rate adjustment value from GIS to simulated monthly temp for each square in grid;
    subtract off ddbase;
    multiply positive temps by days in month and sum up for total growing degrees in the year.
    Compute growing season length.

    Parameters : ddbase -- degree day base
                 monthly_temperature_avgs_lst -- 12 monthly averages for temperature each month
                 monthly_temperature_stds_lst -- 12 monthly standard deviation for how each 
                                             monthly average deviates from year to year
                 monthly_temperature_mins_lst -- 12 min temperature values; 1 for each month from 50+ yrs of 
                                                 daily WMO record of temps in the region
                 monthly_temperature_maxs_lst -- 12 max temperature values; 1 for each month from 50+ yrs of 
                                                 daily WMO record of temps in the region
                 adj_val_mat -- matrix of adjustment values to correct temps based on elev/lapse rate
    Returns : GDD_mat -- a numpy matrix of growing degrees accumulated over the entire year for each plot
              monthly_temp_lst -- a list of 12 matrices of monthly temperatures for each plot
    """
    #### CONSTANTS
    DAYS_IN_MONTH_lst = [31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.]  # days in each month
    monthly_Tmaxs_vec = np.array(monthly_temperature_maxs_lst)
    ####
    def generate_daily_temperatures(month_avg, month_std, num_days):
        # generate X numbers for a gaussian distribution
        return month_std * np.random.randn(num_days) + month_avg

    def generate_month_temperature(month_avg, month_std, minT, maxT, month_number):
        ndays = DAYS_IN_MONTH_lst[month_number]
        daily_temp_vec = generate_daily_temperatures(month_avg, month_std, ndays)
        # if any of the daily values are outside of the range of allowed values, pick new numbers
        # this removes the long tail from the right and left side of the distribution
        while np.any( daily_temp_vec > maxT ) or np.any( daily_temp_vec < minT ):
            new_temps_vec = generate_daily_temperatures(month_avg, month_std, ndays)
            daily_temp_vec[daily_temp_vec > maxT] = new_temps_vec[daily_temp_vec > maxT]
            daily_temp_vec[daily_temp_vec < minT] = new_temps_vec[daily_temp_vec < minT]

        return daily_temp_vec

    # generate a temperature value for each day of the year, but return them in a matrix (MONTH, DAY)
    # Note : some of the returned values will be nan due to different number of days in each month
    def generate_temperatures_matrix(monthly_temperature_avgs_lst, monthly_temperature_stds_lst, 
                                     monthly_minT_vec,  monthly_maxT_vec):
        NMONTH = 12
        NDAY = 31
        # start with all nans and then we will fill in data as we go
        daily_temperature_mat = np.zeros((NMONTH, NDAY)) + np.nan
        month_index = 0
        for month_avg, month_std, month_minT, month_maxT in zip(monthly_temperature_avgs_lst, monthly_temperature_stds_lst, 
                                                                monthly_temperature_mins_lst, monthly_temperature_maxs_lst):
            daily_temp_vec = generate_month_temperature(month_avg, month_std, month_minT, month_maxT, month_index)
            ndays = DAYS_IN_MONTH_lst[month_index]
            daily_temperature_mat[month_index,0:ndays] = daily_temp_vec
            month_index += 1
        return daily_temperature_mat

    # 0) start with the daily temperature values (daily weather for all of geographical grid)
    daily_temperature_mat = generate_temperatures_matrix(monthly_temperature_avgs_lst, monthly_temperature_stds_lst, 
                                                         monthly_temperature_mins_lst, monthly_temperature_maxs_lst)
    ## start geographic specific 
    # for each geographic grid location and each day subtract DDBASE and add the lapse rate adjustment
    # 1) add temp to adj_val for each square in geographic grid; 
    # 2) subtract DDBASE (5.5C) for each square in geographic grid; 
    # 3) sum growing degree days for the year
    nx, ny = lapse_rate_adj_mat.shape
    GDD_mat = np.zeros((nx, ny))
    total_growing_season_mat = np.zeros((nx, ny))
    for x in range(nx):
        for y in range(ny):
            lapse_rate_adj = lapse_rate_adj_mat[x,y]
            growing_degree_mat = daily_temperature_mat - ddbase + lapse_rate_adj
            # every day that is below 0 set to 0
            growing_degree_mat[np.less(growing_degree_mat, 0)] = 0.
            # compute the growing degree days for the year
            growing_degree_days = np.nansum(growing_degree_mat)
            # compute the growing season length as the number of days that have a temperature above DDBASE
            growing_season_ndays = np.sum( np.greater(growing_degree_mat, 0) )
            # store the growing degree days value for this geographic grid point
            GDD_mat[x,y] = growing_degree_days
            total_growing_season_mat[x,y] = growing_season_ndays

    # build up a list by month where each value in the list is a 2D matrix that hold the adjusted month temperature for
    # each point on the geographic grid
    monthly_temp_mat_lst = []
    for month in range(12):
        this_month_avg_temperature = np.nanmean( daily_temperature_mat[month] )
        monthly_temp_mat_lst.append( this_month_avg_temperature + lapse_rate_adj_mat)
    

#    # 1) add temp to adj_val for each square in grid; 2) subtract DDBASE (5.5C); 3) multiply by days in month
#    # 4) sum growing degree days for the year
#    monthly_temp_mat_lst = []
#    GDD_mat = np.zeros(adj_val_mat.shape)
#    growing_season_mat = np.zeros(adj_val_mat.shape)
#    for this_months_temp, days_this_month in zip(month_simtemp_vec, DAYS_IN_MONTH_lst): #this is i,j in zip(a,b)
#        lapse_rate_adj_mat = this_months_temp + adj_val_mat - ddbase
#        lapse_rate_adj_mat[lapse_rate_adj_mat<0] = 0 #sets all negatives to zero, now can sum everything
#        boolean_growing_season_mat = np.where(lapse_rate_adj_mat<=0,0,1) #assigns 0 to every plot where T<ddbase, 1 to every plot where T>ddbase
#        growing_season_this_month_mat = boolean_growing_season_mat * days_this_month
#        growing_season_mat = growing_season_mat + growing_season_this_month_mat #tallies growing days this year for each plot
#        monthly_temp_mat_lst.append(this_months_temp + adj_val_mat)
#        GDD_mat = GDD_mat + lapse_rate_adj_mat * days_this_month #don't use += to increment with numpy!!!

    return GDD_mat, monthly_temp_mat_lst, total_growing_season_mat

'''
def GrowingDegreeDays_calc(ddbase, monthly_temperature_avgs_lst, monthly_temperature_stds_lst, monthly_temperature_maxs_lst, adj_val_mat, month_simtemp_vec=None):
    """
    Simulate monthly temperature based off of driver monthly means and standard deviations;
    add elev/lapse rate adjustment value from GIS to simulated monthly temp for each square in grid;
    subtract off ddbase;
    multiply positive temps by days in month and sum up for total growing degrees in the year.
    Compute growing season length.

    Parameters : ddbase -- degree day base
                 monthly_temperature_avgs_lst -- 12 monthly averages for temperature each month
                 monthly_temperature_stds_lst -- 12 monthly standard deviation for how each 
                                             monthly average deviates from year to year
                 monthly_temperature_maxs_lst -- 12 daily max temperature values from 50+ yrs of 
                                                 WMO record of temps in the region
                 adj_val_mat -- matrix of adjustment values to correct temps based on elev/lapse rate
                 month_simtemp_vec -- optional parameter, if not passed in then generate random temps,
                                      if pass in, this is from an hdf file record
    Returns : GDD_mat -- a numpy matrix of growing degrees accumulated over the entire year for each plot
              monthly_temp_lst -- a list of 12 matrices of monthly temperatures for each plot
    """
    #### CONSTANTS
    DAYS_IN_MONTH_lst = [31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.]  # days in each month
    monthly_Tmaxs_vec = np.array(monthly_temperature_maxs_lst)
    ####
    def generate_temperature(monthly_temperature_avgs_lst, monthly_temperature_stds_lst):
        # simulate temps for each month using staticstics from driver and numpy math
        normal_randn_vec = np.random.randn(12)  # get 12 random numbers with zero mean and std=1
        monthly_avgs_vec = np.array(monthly_temperature_avgs_lst)
        monthly_stds_vec = np.array(monthly_temperature_stds_lst)
        month_simtemp_vec = monthly_avgs_vec + monthly_stds_vec * normal_randn_vec
        return month_simtemp_vec

    if month_simtemp_vec == None:  #generate 12 months of random temperature (weather)
        month_simtemp_vec = generate_temperature(monthly_temperature_avgs_lst, monthly_temperature_stds_lst)
        while np.any(month_simtemp_vec > monthly_Tmaxs_vec):  #if any of 12 values return true, generate 12 new T values
            month_simtemp_vec = generate_temperature(monthly_temperature_avgs_lst, monthly_temperature_stds_lst)
            #while loop will continue until 12 monthly temp values are similated such that all monthlies are less
            #than or equal to the allowed Tmax for that month.

    # 1) add temp to adj_val for each square in grid; 2) subtract 5.5C; 3) multiply by days in month
    # 4) sum growing degree days for the year
    monthly_temp_mat_lst = []
    GDD_mat = np.zeros(adj_val_mat.shape)
    growing_season_mat = np.zeros(adj_val_mat.shape)
    for this_months_temp, days_this_month in zip(month_simtemp_vec, DAYS_IN_MONTH_lst): #this is i,j in zip(a,b)
        lapse_rate_adj_mat = this_months_temp + adj_val_mat - ddbase
        lapse_rate_adj_mat[lapse_rate_adj_mat<0] = 0 #sets all negatives to zero, now can sum everything
        boolean_growing_season_mat = np.where(lapse_rate_adj_mat<=0,0,1) #assigns 0 to every plot where T<ddbase, 1 to every plot where T>ddbase
        growing_season_this_month_mat = boolean_growing_season_mat * days_this_month
        growing_season_mat = growing_season_mat + growing_season_this_month_mat #tallies growing days this year for each plot
        monthly_temp_mat_lst.append(this_months_temp + adj_val_mat)
        GDD_mat = GDD_mat + lapse_rate_adj_mat * days_this_month #don't use += to increment with numpy!!!

    return month_simtemp_vec, GDD_mat, monthly_temp_mat_lst, growing_season_mat
'''
#########################################################################################

def one_time_radiation_readin(monthly_radiation_files_path,expected_nx,expected_ny):
    """
    Activated in year 1 of sim to read-in the radiation files computed in GIS for the simulated terrain;
    generates a list of matrices to be called during PET and soil moisture and light calculations.

    Parameters: monthly_radiation_files_path -- folder location for the 12 monthly radiation matrices
                expected_nx, expected_ny -- define the DEM matrix (obtained from DEM.shape)

    Returns:  radiation_rasters_lst = a list of 12 matrices containing accumulated radiation for each month on each plot
    """
    radiation_rasters_lst = []
    for i in range(12):
        filename = monthly_radiation_files_path+'/monthlyradiation%d.txt' % (i+1)   #rad1.txt corresponds to january.... rad12.txt corresponds to december's radiation
        months_rad_mat = read_in_ascii(filename)
        if months_rad_mat.shape != (expected_nx,expected_ny):
            raise Exception("Monthly radiation file wrong shape: %s" % filename)
        radiation_rasters_lst.append(months_rad_mat)
    return radiation_rasters_lst



    #loop through calling PET function on 12 radiation matricies to get PET for each month
    #compute soil moisture for each month
    #compute the dry days in growing season for the Dry Day Factor constraint on growth




def PET(monthly_temp_mat, rad_raster_mat):
    """
    Using a modified Priestly-Taylor equation as described in Campbell (1977, p140)
    The temperature- and radiation-based PET calculations is recommended for boreal regions by Fisher et al., (2011)
    Simulate monthly PET based on monthly temperature and GIS-computed monthly accumulated radiation (in WH/m2);
    PET is assumed to occur anytime air temperatures are >0C, because conifers can begin transpiration whenever air temp>0C;
    The units work out as follows:
           GIS returns a raster in WH/m2
           total energy = WH/m2 * 3600s/hr = J/m2
           lambda = latent heat of vaporization = 2430 J/g
           a = 0.025 1/deg C
           b = 3 deg C
           PET calculation = a*(avg_monthly_temperature + b)/lambda = g/m2s
           convert to cm/month via = (PET in g/m2s) * (1m3/1000000g) * (100cm/1m)

    Parameters : monthly_temp_PET_lst -- 12 monthly averages for temperature each month
                 
    Returns : ddays = dry day index for each plot, which is a fraction of drought days within growing season
    """
    total_energy = rad_raster_mat * 3600
    monthly_temp_mat[monthly_temp_mat<=0] = -3 #this will result in PET=0 for temps <=0C, at which PET should not be occuring
    PET_mat = (0.025 * (monthly_temp_mat + 3.0) * (total_energy))/(2430.0*10000.0)
    print "PET mat:", PET_mat[0,0]
    return PET_mat

##############################################################################

def rain_sim(rainfall_monthly_avgs, rainfall_monthly_stds):
    """
    Initialize the rain simulator object.

    Parameters: rainfall_monthly_avgs -- 12 monthly averages for rainfall
                rainfall_monthly_stds -- 12 monthly standard deviation for how each 
                                         monthly average deviates from year to year

    Returns : rainfall_vec -- a list of 12 simulated monthly rainfall values for this year of sim
    """
    # store the monthly rainfall statistics
    mean_rain_by_month_vec = np.array(rainfall_monthly_avgs)
    std_rain_by_month_vec = np.array(rainfall_monthly_stds)
    normal_randn_vec = np.random.randn(12)  # get 12 random numbers with zero mean and std=1
    monthly_sim_rain_vec = mean_rain_by_month_vec + std_rain_by_month_vec * normal_randn_vec
    monthly_sim_rain_vec[monthly_sim_rain_vec<0] = 0
    # precipitation is always underestimated due to loss due to winds. According to Bonan, increase the observed rain by 10% (for Alaska boreal)
    # (maybe more for Siberia?) to better represent simulated rain:
    monthly_sim_rain_vec = monthly_sim_rain_vec * 1.1

    print "annual rain: ", np.sum(monthly_sim_rain_vec)
    return monthly_sim_rain_vec



def soil_moisture(monthly_sim_rain, PET_mat, last_months_soil_water_mat, field_capacity, wilting_point):
    """
    Computes soil moisture based on equation:  old_water + rain - PET = current soil moisture
    old water is soil moisture from pervious month

    Parameters: monthly_rain -- simulated precip amount in cm for this month
                PET -- computed in the PET function, this is the list of matrices of PET computed for each plot for each month
                last_months_soil_water -- this is input for the previous month and output for this month

    Returns: last_months_soil_water -- soil moisture computed for this month
    """
    soil_moisture_mat = (last_months_soil_water_mat + monthly_sim_rain) - PET_mat
    runoff_mat = soil_moisture_mat - field_capacity #computing runoff for now, but not using it outside the loop
    runoff_mat[runoff_mat<0] = 0  #sets negative runoff values to zero
    soil_moisture_mat[soil_moisture_mat>field_capacity] = field_capacity  #if more soil moist than field capacity, make that excess run off and set the soil moisture to field capacity (saturated soil)  
    soil_moisture_mat[soil_moisture_mat<(wilting_point - 5.)] = wilting_point - 5. #so soil water can still recharge over the winter, otherwise becomes very negative
#    print "soil water:  ", soil_moisture_mat
    return soil_moisture_mat


def drydays(total_growing_season_mat, soil_moisture_mat, wilting_point, monthly_temp_mat_lst, radiation_mat_lst, field_capacity, monthly_sim_rain_vec, ddbase):
    """
    Take growing season length and soil moisture=f(FC,rain,lastmonthssoilmoist,PET{T&radiation})
    and compute the fraction of dry days within the growing season

    Parameters: total_growing_season_mat -- sum of days within the growing season this year
                soil_moisture_mat -- last month's soil moisture to initate a whole year of soil moisture computations
                wilting point -- specified in the driver, however, allow to dry out below wilting point, unlike ZELIG v1.0
                monthly_temp_mat_lst -- list of matrices of monthly average temperatures for each plot for each month
                radiation_mat_lst -- list of matrices of monthly cumulative incident radiation for each plot
                field_capacity -- specified in driver, soil moisture in excess of this is considered runoff
                monthly_sim_rain_vec -- list of monthly precip (each plot in simulated area is considered to receive same amount of precip, due to how small the 
                                        simulated area is)

    Returns: soil_moisture_mat = december's soil moisture from this year to be used to initiate the soil moisture computation next year
             drydays_fraction_mat =  = a fraction of growing season spent in drought this year (0 to 1)
    """
    days_this_month_lst = [31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.]  # days in each month    
    dry_days_accumulator_mat = np.zeros(total_growing_season_mat.shape)
    for month in range(12):
        PET_mat = PET(monthly_temp_mat = monthly_temp_mat_lst[month], rad_raster_mat = radiation_mat_lst[month])
        # since actual evapotranspiration (AET) is about 70% of PET, scale down the computed PET:
        PET_mat = 0.7 * PET_mat
        soil_moisture_mat = soil_moisture(monthly_sim_rain = monthly_sim_rain_vec[month], PET_mat = PET_mat, 
                                          last_months_soil_water_mat = soil_moisture_mat, 
                                          field_capacity = field_capacity, wilting_point = wilting_point)
        boolean_drought_mat = np.where(((soil_moisture_mat<=wilting_point) & (monthly_temp_mat_lst[month]>=ddbase)),1,0) #assigns 0 to every plot where there is less soil water than wilting point and within growing season
        drought_this_month_mat = boolean_drought_mat * days_this_month_lst[month] # number of days in dought
        dry_days_accumulator_mat = dry_days_accumulator_mat + drought_this_month_mat #tallies drought days this year for each plot
#        print "PET monthly sum for all plots:  ", PET_mat.sum(), "monthly rain summed up over all plots:  ", monthly_sim_rain_vec[month]*900.
    drydays_fraction_mat = dry_days_accumulator_mat/total_growing_season_mat
        
#    print "rain:  ", monthly_sim_rain_vec
#    print "dry days fraction:  ", drydays_fraction_mat
    return soil_moisture_mat, drydays_fraction_mat


######################################################################################################

#if __name__ == '__main__':
def compute_GDD():
    from load_driver import load_driver_json
    driver_file = 'driver_boreal.json'   #for testing, comparing against ZELIG v1.0 from Urban 1990
    # load the species specific parameters from the driver file into a dictionary called driver
    driver = load_driver_json(driver_file)
    # define the range of years to simulate over
    start = 0; stop = driver["NYRS"]-1
    nplots = driver["NPLOTS"]
    GDD_matrix, monthly_temp_mat_lst, total_growing_season_mat = GrowingDegreeDays_calc(ddbase = 5.5, monthly_temperature_avgs_lst = driver["XT"], 
                                                                                  monthly_temperature_stds_lst = driver["VT"], 
                                                                                  lapse_rate_adj_mat = read_in_ascii('elev_adj_factor.txt'))
    generate_ERDASimg_grid(metadata_file = 'elev_adj_factor.txt', matrix_file = 'GDD_grid.img',
                        numpy_raster = GDD_matrix)
    
    radiation_mat_lst = one_time_radiation_readin()
    monthly_sim_rain_vec = rain_sim(rainfall_monthly_avgs = driver['XR'], rainfall_monthly_stds = driver['VR'])
    initial_soil_water_mat = np.zeros(GDD_matrix.shape)
    initial_soil_water_mat = driver['FC'] #start sim with FC as soil water content
    soil_moisture_mat, drydays_fraction_mat = drydays(total_growing_season_mat = total_growing_season_mat,soil_moisture_mat = initial_soil_water_mat, 
                                                      wilting_point = driver['WP'], monthly_temp_mat_lst = monthly_temp_mat_lst, 
                                                      radiation_mat_lst = radiation_mat_lst, field_capacity = driver['FC'],
                                                      monthly_sim_rain_vec = monthly_sim_rain_vec, ddbase = 5.56) #specify ddbase in driver?
    generate_ERDASimg_grid(metadata_file = 'elev_adj_factor.txt', matrix_file = 'DryDays_grid.img',
                        numpy_raster = drydays_fraction_mat)
