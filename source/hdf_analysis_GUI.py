"""
Simple pyqtgraph application to investigate a SiBork driver file.
"""
import pyqtgraph as pg
import pyqtgraph.opengl as pggl
import pyqtgraph.console
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import h5py
import dill
from math import *
import pandas as pd
import StringIO
from hdf_analysis import load_driver, compute_results_biovolume, compute_results_biomass, \
                         compute_results_leaf_area, compute_results_basal_area, compute_results_stems, \
                         compute_results_average_dbh, compute_results_average_height, \
                         compute_results_loreys_height



#@numba.jit('float64[:](float64[:], float64)')
def custom_pseudoScatter(data, spacing):
    """
    Used for examining the distribution of values in a set. Produces scattering as in beeswarm or column scatter plots.
    
    Given a list of x-values, construct a set of y-values such that an x,y scatter-plot
    will not have overlapping points (it will look similar to a histogram).
    """
    #s2 = spacing*5
    s2 = 1.
    xmin = min(data)
    xmax = max(data) #+1e-6
    NBINS = 100
    xstep = (xmax - xmin) / NBINS
    ymax_vec = np.zeros(NBINS+1)
    yresults_vec = np.zeros(len(data))

    if xstep > 0:
        for i in range(len(data)):
            x = data[i]
            index = int( (x - xmin)/xstep )
            y = ymax_vec[index] + s2
            ymax_vec[index] += s2
            yresults_vec[i] = y
        return yresults_vec
    else:
        for i in range(len(data)):
            x = data[i]
            index = int(x)
            y = ymax_vec[index] + s2
            ymax_vec[index] += s2
            yresults_vec[i] = y
        return yresults_vec


def make_colors(num_colors):
    color_tuples = [
                   (255,0,0),     # red
                   (0,255,0),     # green
                   (0,0,255),     # blue
                   (100,100,100), # grey
                   (250,10,204),  # magenta
                   (243,168,11),  # orange
                   (23,231,253),  # cyan
                   (162,69,199),  # purple
                   (249,255,13),  # yellow
                   (109,64,37),   # brown
                   (75,193,148),  # teal
                   ]
    if num_colors > len(color_tuples):
        raise Exception('TODO: define more colors in function make_colors.')
    return [pg.mkColor(color_tuples[i]) for i in range(num_colors)]

    
###########################################################################3

def display(command):
    custom_commands = ['DEM',]
    if command == '?':
        return 'custom commands : %s\n' % custom_commands
    elif command in custom_commands:
        if command == 'DEM':
            display_DEM()
        return 'custom display'


def swarm(command):
    """
    Create a timeseries animation of species-specific variables, i.e. how does individual tree biovolume change over time (color-coded by species)
    'dbh': diameter-at-breast-height (cm) for each tree in the simulation (each tree followed through time, until mortality)
           example: swarm('dbh')
    'bv':  biovolume (m3) for each tree in the simulation (each tree followed through time, until mortality)
           example: swarm('bv')
    'rsf': relative soil fertility for each plot over time, nothing to do with species-specific nutri factor, 
           just the fraction of optimal biomass that can be added to each plot each year (based on site index)
           example: swarm('rsf')
    'ht':  height (m)of the tallest tree on each plot
           example: swarm('ht')
    """
    global timer, timer_calls
    commands = ['dbh','bv','rsf','ht']
    if command in commands:
        if command == 'dbh':
            display_swarm_dbh()
        elif command == 'bv':
            display_swarm_biovolume()
        elif command == 'rsf':
            display_swarm_relative_soil_fertility()
        elif command == 'ht':
            display_swarm_max_height()
        timer.start(200)

def equation(command):
    """
    Create a of species-specific allometric equations for DBHs 0-max
    'ht':  what the height (m) for each species would be based on a given diameter
           example: equation('ht')
    'bv':  biovolume (m3) for a tree of a given species for a given DBH
    'bm':  biomass (t=Mg) for a tree of a given species for a given DBH
    'la':  leaf area (m2) for a tree of a given species for a given DBH (this is computed based on foliarBiomass*specificLeafArea)
    'ba':  basal area (m2) for a tree of a given DBH (same equation for all spp)
    'oi':  species-specific optimal diameter increment (cm) for max tree growth without any environmental limitations
    'sff': soil fertility tolerances (3 NUTRI classes)
    'smf': drought tolerance (5 classes)
    'alf': shade tolerance (5 classes)
    'ddf': heat tolerance (based on either a parabolic or a saturation curve, depending on which is activated in the driver)
    """
    def launch(command):
        if command == 'ht':
            display_equation_height()
        elif command == 'bv':
            display_equation_biovolume()
        elif command == 'bm':
            display_equation_biomass()
        elif command == 'la':
            display_equation_leaf_area()
        elif command == 'ba':
            display_equation_basal_area()
        elif command == 'oi':
            display_equation_optimal_growth_increment()
        elif command == 'sff':
            display_equation_soil_fertility_factor()
        elif command == 'smf':
            display_equation_soil_moisture_factor()
        elif command == 'alf':
            display_equation_available_light_factor()
        elif command == 'ddf':
            display_equation_degree_day_factor()
    equation_commands = ['ht','bv','bm','la','ba','oi','sff','smf','alf','ddf']
    if command == 'all':
        for cmd in equation_commands:
            launch(cmd)
    elif command in equation_commands:
        launch(command)
    else:
        print 'Invalid command :: equation("%s")' % command

def results(command, **kwargs):
    """
    Show model output by species (this may be spatially averaged within a simulation, or an average of multiple replicates, depending on what the HDF file contains)
    The coolest thing is that you can exclude small DBHs from the averaging by specifying the min_dbh value.
    'bv':  average total biovolume (m3/ha) per plot for each species, normalized to per hectare
           example: results('bv')
           if interested only in certain areas, e.g. where warming of +2C was applied, specify mask
           example: mask=(driver['elevation_lapse_rate_adjustment_matrix']==2)
                    results('bv',mask=mask)
    'bm':  average total biomass (t/ha=Mg/ha) per plot for each species, normalized to per hectare
    'la':  average total leaf area (m2/ha) per plot for each species, normalized to per hectare
    'ba':  average total basal area (m2/ha) per plot for each species, normalized to per hectare
    'ht':  average height (m) for a tree of each species, averaged across multiple plots
    'stems':  average number of stems of each species on each plot, normalized to per hectare
    'dbh': average DBH (cm) for each species
           example: results('dbh',min_dbh=5.0)   #this will exclude small saplings with DBH<5.0cm from the average
    'ht':  average tree height (m) for each species
    'lht': Lorey's height (m) for each species (contribution of height to the average is weighted by the tree's basal area)
    """
    def launch(command, **kwargs):
        if command == 'bv':
            display_results_biovolume(**kwargs)
        elif command == 'bm':
            display_results_biomass(**kwargs)
        elif command == 'la':
            display_results_leaf_area(**kwargs)
        elif command == 'ba':
            display_results_basal_area(**kwargs)
        elif command == 'stems':
            display_results_stems(**kwargs)
        elif command == 'dbh':
            display_results_average_dbh(**kwargs)
        elif command == 'ht':
            display_results_average_height(**kwargs)
        elif command == 'lht':
            display_results_loreys_height(**kwargs)
    commands = ['bv','bm','la','ba','stems','dbh','ht','lht']
    if command == 'all':
        for cmd in commands:
            launch(cmd, **kwargs)
    elif command in commands:
        launch(command, **kwargs)
    else:
        print 'Invalid command :: results("%s")' % command

def weather(command, **kwargs):
    """
    Check out the weather effects on vegetation through growing degree days and faction of growing season in drought (below wilting point).
    'gdd':  annual tally of growing degree days (above DDBASE specified in driver)
           if interested only in certain areas, e.g. where warming of +2C was applied, specify mask
           example: mask=(driver['elevation_lapse_rate_adjustment_matrix']==2)
                    weather('gdd',mask=mask)
    'dry':  annual value for the fraction of the growing season during which soil moisture is at or below wilting point (15bar)
            example: weather('dry')
    """
    def launch(command, **kwargs):
        if command == 'gdd':
            display_weather_degree_days(**kwargs)
        elif command == 'dry':
            display_weather_dry_days(**kwargs)
    commands = ['gdd','dry',]
    if command == 'all':
        for cmd in commands:
            launch(cmd, **kwargs)
    elif command in commands:
        launch(command, **kwargs)
    else:
        print 'Invalid command :: weather("%s")' % command

def factor(command, **kwargs):
    """
    Summary of how the different environmental factors have been effecting each species each year of the simulation.
    'ddf':  value 0 to 1. Growing degree days (above DDBASE specified in driver) are a plot-wide parameter. Trees are affected on a per-species basis (mostly depends on DDMIN requirement).
            example: factor('ddf')
           if interested only in certain areas, e.g. where warming of +2C was applied, specify mask
           example: mask=(driver['elevation_lapse_rate_adjustment_matrix']==2)
                    factor('ddf',mask=mask)
    'smf':  value 0 to 1. Soil moisture is a simulation-wide parameter, same soil type is assumed across the entire simulation area, so same field capacity and wilting point. However,
            different species are affected differently by the soil moisture based on drought tolerances set in driver.
    'sff':  value 0 to 1. soil fertility is set by site index at the plot-level. Different species have different soil nutrition requirements, as set in the driver. When the possible annual
            biomass increment computed using alf*ddf*smf exceeds the site-index-related max amount of annual growth that can be supported on this soil, all species are throttled back, 
            but some species are more sensitive to this limitation and so their soil fertility factor is decreased more, which will affect overall growth if soil moisture is not (as) limiting.
    'alf':  value 0 to 1. This factor is specific to where the tree is located within the canopy on a plot. It is computed based on the 3-D light subroutine for each vertical meter of canopy, 
            and then averaged across the canopy of each tree. Then, it is scaled by the tree's shade tolerance class set in the driver.
    'growth':  value 0 to 1. This is the overall environmental effect on tree growth = alf*ddf*min(smf,sff), however, sff is only limiting if other factors are not as limiting (see sff doc string)
    """
    def launch(command, **kwargs):
        if command == 'ddf':
            display_factor_growing_degree_days(**kwargs)
        elif command == 'smf':
            display_factor_soil_moisture(**kwargs)
        elif command == 'sff':
            display_factor_soil_fertility(**kwargs)
        elif command == 'alf':
            display_factor_available_light(**kwargs)
        elif command == 'growth':
            display_factor_growth(**kwargs)
    commands = ['ddf','smf','sff','alf','growth']
    if command == 'all':
        for cmd in commands:
            launch(cmd, **kwargs)
    elif command in commands:
        launch(command, **kwargs)
    else:
        print 'Invalid command :: factor("%s")' % command


def spatial(command):
    """
    The tallest tree on each plot is represented with a pillar with the height proportional to tree height, so each plot is only represented by the tallest tree on that plot. 
    The pillars are color-coded by species. As trees grow and compete, the dominances on each plot (as assessed through height) changes, so pillars change height and color. 
    Only shows the pillars on a flat plane, so if simulated 3-D terrain, can see how the heights of tallest trees on each plot are relative to each other, but the terrain is subtracted out.
    example: spatial('ht')
    TODO: should dominance be determined by largest biovolume instead of greatest height?
    """
    def launch(command):
        if command == 'ht':
            display_spatial_max_height()
    commands = ['ht',]
    if command == 'all':
        for cmd in commands:
            launch(cmd)
        timer.start(200)
    elif command in commands:
        launch(command)
        #display_swarm_dbh()  #uncomment to have the dbh swarm window pop up whenever the spatial tree trunk bar graph is generated
        timer.start(200)
    else:
        print 'Invalid command :: spatial("%s")' % command

def scatter(command, **kwargs):
    """
    Animates how the different environmental factors change for each species over time.
    'ddf':  Ddegree days factor is the same for all individuals of a given species for simulation on flat terrain. In complex terrain, this represents the spatial average for the species.
            example: scatter('ddf')
            if interested only in certain areas, e.g. where warming of +2C was applied, specify mask
            example: mask=(driver['elevation_lapse_rate_adjustment_matrix']==2)
                     scatter('ddf',mask=mask)
    'smf':  Soil moisture factor is the same for all individuals of a given species in the simulation while no runon is implemented.
    'sff':  Soil fertility factor is the same for all individual of a given species within a site index block. Across a gradient, this represents the spatial average for the species.
    'alf':  CANNOT BE IMPLEMENTED, because the available light factor is on a per-tree basis and depends on where the tree is in the canopy. An average of this for a species would make no sense,
            because trees of the same species may be experiencing extremely different conditions within the canopy even though the have the same tolerances.
    """
    def launch(command, **kwargs):
        if command == 'ddf':
            display_scatter_factor_growing_degree_days(**kwargs)
        elif command == 'smf':
            display_scatter_factor_soil_moisture(**kwargs)
        elif command == 'sff':
            display_scatter_factor_soil_fertility(**kwargs)
    commands = ['ddf','smf','sff']
    if command == 'all':
        for cmd in commands:
            launch(cmd, **kwargs)
        timer.start(200)
    elif command in commands:
        launch(command, **kwargs)
        timer.start(200)
    else:
        print 'Invalid command :: scatter("%s")' % command

def make_raster(command, **kwargs):
    """
    Make a georeferenced ASCII file to be read into GIS for
    'spp': the dominant species on each plot by max biomass
           example: make_raster('spp',year=100,raster_filename='gisspp.txt')
    'lht': Lorey's height on each plot
           example: make_raster('lht',year=100,raster_filename='gislht.txt')
    'bm': total biomass from each plot
           example: make_raster('bm',year=100,raster_filename='gisbm.txt')
    """
    def launch(command, **kwargs):
        if command == 'spp':
            make_spp_raster(**kwargs)
#        elif command == 'lht':
#            make_lht_raster(**kwargs)
#        elif command == 'bm':
#            make_bm_raster(**kwargs)
    commands = ['spp','lht','bm']
    if command == 'all':
        for cmd in commands:
            launch(cmd, **kwargs)
        timer.start(200)
    elif command in commands:
        launch(command, **kwargs)
        timer.start(200)
    else:
        print 'Invalid command :: make_raster("%s")' % command

def make_spp_raster(year, raster_filename, **kwargs):
    """
    Mask each spp # and loop through vectorizing to compute biomass for each tree on plot. Then mask by max on biomass and spp_code arrays.
    """
    def determine_dominance(year,func_key):  #return an array of plot-level results for the specified year, func_key can be 'BIOMASS_EQUATION', 'HEIGHT_EQUATION', 'BASAL_AREA_EQUATION', etc/
        year_str = '%.4d' % year
        # pull the current year dbh matrix from the hdf file
        dbh_matrix = np.array(h5file['DBH'][year_str])
        nx,ny,ntrees = dbh_matrix.shape
        # pull the current year species code matrix from the hdf file
        species_code_matrix = np.array(h5file['SpeciesCode'][year_str])
        num_species = len(driver['species_code_to_name'])
        results_matrix = np.zeros((nx,ny))

        for x in range(nx):
            for y in range(ny):
                biggest_tree = 0
                biggest_tree_species_code = 0
                # compute the species specific height values for each species
                for current_species_code in range(num_species):
                    # get all of the dbh values for the species code of interest
                    dbh_vec = dbh_matrix[x,y][species_code_matrix[x,y] == current_species_code]
                    if np.any(dbh_vec):
                        # convert dbh into height
                        species_name = driver['species_code_to_name'][current_species_code]
                        fn = driver['species'][species_name][func_key]
                        results_vec = fn(dbh_vec)
                        biggest_of_this_species = np.max(results_vec)
                        if biggest_of_this_species > biggest_tree:
                            biggest_tree = biggest_of_this_species
                            biggest_tree_species_code = current_species_code
                results_matrix[x,y] = biggest_tree_species_code
        print results_matrix.shape
        return results_matrix

    #generate the plot-level matrix of spp for largest tree by biomass on plot
    biomass_raster_mat = determine_dominance(year,func_key='BIOVOLUME_EQUATION')
    create_gis_raster(biomass_raster_mat, raster_filename)

def create_gis_raster(matrix, raster_filename):
    #Write the plot-level output matrix
    ncols = driver['NS_number_of_plots']
    nrows = driver['EW_number_of_plots']
    NWx, NWy = driver['north_west_corner_coordinates']
    cellsize = driver['EW_plot_length_m']
    xllcorner = NWx
    yllcorner = NWy - cellsize*nrows
    NODATA_value = '-9999'
    georef_header = \
"""ncols         %d
nrows         %d
xllcorner     %f
yllcorner     %f
cellsize      %d
NODATA_value  -9999
""" % (ncols,nrows,xllcorner,yllcorner,cellsize)
    dataframe = pd.DataFrame(matrix)
    output = StringIO.StringIO()
    dataframe.to_csv(path_or_buf=output, sep=' ', index=False, header=False)
    ascii_raster = output.getvalue()
    full_ascii = georef_header + ascii_raster
    f=open(raster_filename,'w')
    f.write(full_ascii)
    f.close()

def make_lht_raster(year, **kwargs):
    pass

def make_bv_raster(year, **kwargs):
    pass

def display_equation_height():
    display_fns_of_dbh(title="Species Specific Height Equations",
                       left_label='Height (m)',
                       fn_key="TREE_HEIGHT_EQUATION")

def display_equation_biovolume():
    display_fns_of_dbh(title="Species Specific BioVolume Equations",
                       left_label='BioVolume (m^3)',
                       fn_key="BIOVOLUME_EQUATION")

def display_equation_biomass():
    display_fns_of_dbh(title="Species Specific Biomass Equations",
                       left_label='Biomass (Mg)',
                       fn_key="BIOMASS_EQUATION")

def display_equation_leaf_area():
    display_fns_of_dbh(title="Species Specific Leaf Area Equations",
                       left_label='Leaf Area (m^2)',
                       fn_key="LEAF_AREA_EQUATION")

def display_equation_basal_area():
    display_fns_of_dbh(title="Species Specific Basal Area Equations",
                       left_label='Basal Area (m^2)',
                       fn_key="BASAL_AREA_EQUATION")
    
def display_equation_optimal_growth_increment():
    display_fns_of_dbh(title="Species Specific Optimal Growth Increment Equations", 
                       left_label='increment (cm)', 
                       fn_key="OPTIMAL_GROWTH_INCREMENT_EQUATION")

# generalize plot that plot over dbh
def display_fns_of_dbh(title, left_label, fn_key):
    print 'opening ', title
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom='dbh (cm)', left=left_label)
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species+1)]
    colors = make_colors(num_species)

    for current_species_code in range(num_species):
        species_name = driver['species_code_to_name'][current_species_code]
        fn = driver['species'][species_name][fn_key]
        DMAX = driver['species'][species_name]['DMAX']

        # add a species specific line to the plot
        xs = np.arange(0.01, DMAX, 0.1)
        ys = fn(xs)
        curve = plot.plot(xs, ys, pen=colors[current_species_code])

        ## Create text object, use HTML tags to specify color/size
        #text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">%s</span></div>' % species_name,
        #                   anchor=(-0.3,1.3)) #, border='w', fill=(0, 0, 255, 100))
        text = pg.TextItem(species_name, anchor=(-0.3,1.3), color=colors[current_species_code])
        plot.addItem(text)
        text.setPos(max(xs), ys[-1])
        ## Draw an arrowhead next to the text box
        arrow = pg.ArrowItem(pos=(max(xs), ys[-1]), angle=-45, headLen=15)
        plot.addItem(arrow)

    plot.show()


def display_equation_soil_fertility_factor():
    return display_fns_of_relative_to_1(title='Species Specific Soil Fertility Factor', 
                                        bottom_label='relative soil fertility', 
                                        left_label='factor (relative to 1)',
                                        xlabel_pos=0.2,
                                        fn_key='SOIL_FERTILITY_FACTOR_EQUATION')

def display_equation_soil_moisture_factor():
    return display_fns_of_relative_to_1(title='Species Specific Soil Moisture Factor', 
                                        bottom_label='relative dry days', 
                                        left_label='factor (relative to 1)',
                                        xlabel_pos=0.05,
                                        fn_key="SOIL_MOISTURE_FACTOR_EQUATION")

def display_equation_available_light_factor():
    return display_fns_of_relative_to_1(title='Species Specific Available Light Factor', 
                                        bottom_label='relative available light', 
                                        left_label='factor (relative to 1)',
                                        xlabel_pos=0.6,
                                        fn_key="AVAILABLE_LIGHT_FACTOR_EQUATION")


# generalize plot that plot over an x range of 0 to 1
def display_fns_of_relative_to_1(title, bottom_label, left_label, xlabel_pos, fn_key):
    print 'opening ', title
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species+1)]
    colors = make_colors(num_species)
    for current_species_code in range(num_species): 
        species_name = driver['species_code_to_name'][current_species_code]
        fn = driver['species'][species_name][fn_key]

        # add a species specific line to the plot
        xs = np.arange(0.0, 1.0, 0.01)
        ys = fn(xs)
        #ys = np.array([fn(x) for x in xs])
        curve = plot.plot(xs, ys, pen=colors[current_species_code])

        # place the labels mid-way through the curve
        xpos = xlabel_pos
        ypos = ys[xs == xpos]
        ## Create text object, use HTML tags to specify color/size
        #text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">%s</span></div>' % species_name,
        #                   anchor=(-0.3,1.3)) #, border='w', fill=(0, 0, 255, 100))
        text = pg.TextItem(species_name, anchor=(-0.3,1.3), color=colors[current_species_code])
        plot.addItem(text)
        text.setPos(xpos, ypos)
        ## Draw an arrowhead next to the text box
        arrow = pg.ArrowItem(pos=(xpos, ypos), angle=-45, headLen=15)
        plot.addItem(arrow)

    plot.show()
    return plot


def display_equation_degree_day_factor(title='Species Specific Degree Day Factor',
                                       bottom_label='degree days',
                                       left_label='factor (relative to 1)',
                                       fn_key='DEGREE_DAY_FACTOR_EQUATION'):
    print 'opening ', title
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species+1)]
    colors = make_colors(num_species)
    for current_species_code in range(num_species): 
        species_name = driver['species_code_to_name'][current_species_code]
        fn = driver['species'][species_name][fn_key]

        # add a species specific line to the plot
        xs = np.arange(100, 3000, 10)
        ys = fn(xs)
        curve = plot.plot(xs, ys, pen=colors[current_species_code])

        # place the labels mid-way through the curve
        ypos = max(ys)
        xpos = xs[ys == ypos][0]
        ## Create text object, use HTML tags to specify color/size
        #text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">%s</span></div>' % species_name,
        #                   anchor=(-0.3,1.3)) #, border='w', fill=(0, 0, 255, 100))
        text = pg.TextItem(species_name, anchor=(-0.3,1.3), color=colors[current_species_code])
        plot.addItem(text)
        text.setPos(xpos, ypos)
        ## Draw an arrowhead next to the text box
        arrow = pg.ArrowItem(pos=(xpos, ypos), angle=-45, headLen=15)
        plot.addItem(arrow)

    plot.show()
    return plot

# global variables used by the animations
#year=1; stop_year=100
timer_calls = []
timer = QtCore.QTimer()
def update_timer():
    global timer, timer_calls
    prune_list = []
    if timer_calls:
        for ani_fn in timer_calls:
            ani_fn.call()
            if not ani_fn.running:
                prune_list.append(ani_fn)
        # prune any dead animation callbacks
        for obj in prune_list:
            timer_calls.remove(obj)
    else:
        # no callbacks to run, so stop the timer
        timer.stop()
timer.timeout.connect(update_timer)
    
class AnimationCallback(object):
    def __init__(self, years_in_sim, fn):
        self.current_year_index = 0
        self.years_in_sim = years_in_sim
        self.fn = fn
        self.running = True
    def call(self):
        if self.running:
            year = self.years_in_sim[self.current_year_index]
            self.fn(year)
            self.current_year_index += 1
            if self.current_year_index >= len(self.years_in_sim):
                self.running = False
        

SWARM_SAMPLE_EDGE = 8

def display_swarm_max_height(title='Swarm Maximum Tree Height per Plot',
                             bottom_label='height (m)',
                             left_label='occurence'):
    global timer_calls
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    dmax_by_species = [driver['species'][species]['DMAX'] for species in driver['species']]
    height_fn_by_species = [driver['species'][species]['TREE_HEIGHT_EQUATION'] for species in driver['species']]
    max_height = max([fn(dmax) for fn, dmax in zip(height_fn_by_species, dmax_by_species)])
    plot.setRange(xRange=(0,max_height), yRange=(0,200))
    legend = plot.addLegend()
    num_species = len(driver['species_code_to_name'])
    colors = make_colors(num_species)
    first_run = [True for species in range(num_species)]  # trick use a mutable list to be able to write to this closure captured variable : python3 uses nonlocal keyword instead
    # the update function that will be called by the timer
    def update(year):
        year_str = '%.4d' % year
        # pull the current year dbh matrix from the hdf file
        dbh_matrix = np.array(h5file['DBH'][year_str]) #[0:SWARM_SAMPLE_EDGE,0:SWARM_SAMPLE_EDGE])
        # pull the current year species code matrix from the hdf file
        species_code_matrix = np.array(h5file['SpeciesCode'][year_str]) #[0:SWARM_SAMPLE_EDGE,0:SWARM_SAMPLE_EDGE])

        # clear the previous plot
        plot.clear()
        nx,ny,ntrees = dbh_matrix.shape
        height_matrix = np.zeros((nx,ny)) + np.nan
        tallest_species_code_matrix = np.zeros((nx,ny)) + np.nan
        for x in range(nx):
            for y in range(ny):
                tallest_tree = 0
                tallest_tree_species_code = np.nan
                # compute the species specific height values for each species
                for current_species_code in range(num_species):
                    # get all of the dbh values for the species code of interest
                    dbh_vec = dbh_matrix[x,y][species_code_matrix[x,y] == current_species_code]
                    if np.any(dbh_vec):
                        # convert dbh into height
                        species_name = driver['species_code_to_name'][current_species_code]
                        fn = driver['species'][species_name]['TREE_HEIGHT_EQUATION']
                        height_vec = fn(dbh_vec)
                        tallest_of_this_species = np.max(height_vec)
                        if tallest_of_this_species > tallest_tree:
                            tallest_tree = tallest_of_this_species
                            tallest_tree_species_code = current_species_code
                if np.isnan(tallest_tree_species_code):
                    tallest_tree = np.nan
                # store the height and species code of the tallest tree on this plot
                height_matrix[x,y] = tallest_tree
                tallest_species_code_matrix[x,y] = tallest_tree_species_code

        for current_species_code in range(num_species):
            species_name = driver['species_code_to_name'][current_species_code]
            # get all of the height values for the species code of interest
            data_vec = height_matrix[tallest_species_code_matrix == current_species_code]

            if np.any(data_vec):
                ### update the swarm histogram plot
                ## Now draw all points as a nicely-spaced scatter plot (swarm)
                y = custom_pseudoScatter(data_vec, spacing=0.15)  # the combination of x=vals & y=histogram height makes a nice swarm scatterplot that resembles a histogram
                if first_run[current_species_code]:
                    plot.plot(data_vec, y, pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40), name=species_name)
                    first_run[current_species_code] = False
                else:
                    # to keep the legend from growing
                    plot.plot(data_vec, y, pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40),)
        if np.any(np.isnan(height_matrix)):
            number_empty_plots = np.sum(np.isnan(height_matrix))
            print 'year %s : %s empty plots' %(year, number_empty_plots)
        plot.setTitle('%s : year : %s' % (title, year))

    years_to_run_list = driver['simulation_years_logged']
    callback_obj = AnimationCallback(years_to_run_list, update)
    timer_calls.append(callback_obj)

def display_swarm_relative_soil_fertility(title='Swarm Relative Soil Fertility',
                                          bottom_label='Relative Fertility (0 to 1)',
                                          left_label='occurence'):
    global timer_calls
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    plot.setRange(xRange=(0,1), yRange=(0,200))
    # the update function that will be called by the timer
    def update(year):
        year_str = '%.4d' % year
        # pull the current year relative soil fertility matrix
        relative_soil_fertility_matrix = np.array(h5file['RelativeSoilFertility'][year_str])
        data_vec = relative_soil_fertility_matrix.reshape(-1)

        # clear the previous plot
        plot.clear()
        ### update the swarm histogram plot
        ## Now draw all points as a nicely-spaced scatter plot (swarm)
        y = custom_pseudoScatter(data_vec, spacing=0.15)  # the combination of x=vals & y=histogram height makes a nice swarm scatterplot that resembles a histogram
        plot.plot(data_vec, y, pen=None, symbol='o', symbolSize=5) #, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40),)
        plot.setTitle('%s : year : %s' % (title, year))

    years_to_run_list = driver['simulation_years_logged']
    callback_obj = AnimationCallback(years_to_run_list, update)
    timer_calls.append(callback_obj)


def display_swarm_dbh(title='Swarm Histogram of DBH',
                      bottom_label='dbh (cm)',
                      left_label='occurence'):
    global timer_calls
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    max_diameter = max([driver['species'][species]['DMAX'] for species in driver['species']]) + 10
    plot.setRange(xRange=(0,max_diameter), yRange=(0,200))
    legend = plot.addLegend()
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species)]
    colors = make_colors(num_species)
    first_run = [True for species in range(num_species)]  # trick use a mutable list to be able to write to this closure captured variable : python3 uses nonlocal keyword instead
    # the update function that will be called by the timer
    def update(year):
        year_str = '%.4d' % year
        # pull the current year dbh matrix from the hdf file
        dbh_matrix = np.array(h5file['DBH'][year_str][0:SWARM_SAMPLE_EDGE,0:SWARM_SAMPLE_EDGE])
        # pull the current year species code matrix from the hdf file
        species_code_matrix = np.array(h5file['SpeciesCode'][year_str][0:SWARM_SAMPLE_EDGE,0:SWARM_SAMPLE_EDGE])

        # clear the previous plot
        plot.clear()
        for current_species_code in range(num_species):
            species_name = driver['species_code_to_name'][current_species_code]
            # get all of the dbh values for the species code of interest
            dbh_vec = dbh_matrix[species_code_matrix == current_species_code]
            print '%s : %d trees' %(species_name, len(dbh_vec))

            if np.any(dbh_vec):
                ### update the swarm histogram plot
                ## Now draw all points as a nicely-spaced scatter plot (swarm)
                y = custom_pseudoScatter(dbh_vec, spacing=0.15)  # the combination of x=vals & y=histogram height makes a nice swarm scatterplot that resembles a histogram
                if first_run[current_species_code]:
                    plot.plot(dbh_vec, y, pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40), name=species_name)
                    first_run[current_species_code] = False
                else:
                    # to keep the legend from growing
                    plot.plot(dbh_vec, y, pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40),)
        plot.setTitle('%s : year : %s' % (title, year))

    years_to_run_list = driver['simulation_years_logged']
    callback_obj = AnimationCallback(years_to_run_list, update)
    timer_calls.append(callback_obj)


def display_swarm_biovolume(title='Swarm Histogram of Biovolume',
                            bottom_label='Biovolume (m^3/tree)',
                            left_label='occurence'):
    global timer_calls
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    plot.setRange(xRange=(0,5), yRange=(0,60))
    legend = plot.addLegend()
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species)]
    colors = make_colors(num_species)
    first_run = [True for species in range(num_species)]  # trick use a mutable list to be able to write to this closure captured variable : python3 uses nonlocal keyword instead
    # the update function that will be called by the timer
    def update(year):
        year_str = '%.4d' % year
        # pull the current year dbh matrix from the hdf file
        dbh_matrix = np.array(h5file['DBH'][year_str][0:SWARM_SAMPLE_EDGE,0:SWARM_SAMPLE_EDGE])
        # pull the current year species code matrix from the hdf file
        species_code_matrix = np.array(h5file['SpeciesCode'][year_str][0:SWARM_SAMPLE_EDGE,0:SWARM_SAMPLE_EDGE])
        ## for speed reasons, we only took a portion of the dbh matrix that is available
        ## the matrix should be nx,ny,ntrees in size
        #nx, ny, ntrees = dbh_matrix.shape
        ## compute the sample area
        #sample_area_m2 = nx * ny * driver['plot_area']
        #SQ_METERS_PER_HA = 100 * 100
        #sample_area_ha = sample_area_m2 / SQ_METERS_PER_HA

        # clear the previous plot
        plot.clear()
        for current_species_code in range(num_species):
            species_name = driver['species_code_to_name'][current_species_code]
            # get all of the dbh values for the species code of interest
            dbh_vec = dbh_matrix[species_code_matrix == current_species_code]
            # compute the species specific biovolume
            fn = driver['species'][species_name]["BIOVOLUME_EQUATION"]
            biovolume_vec = fn(dbh_vec)  # use numba generated ufunc
            #biovolume_vec = np.array([fn(dbh) for dbh in dbh_vec])

            if np.any(dbh_vec):
                ### update the swarm histogram plot
                ## Now draw all points as a nicely-spaced scatter plot (swarm)
                y = custom_pseudoScatter(biovolume_vec, spacing=0.15)  # the combination of x=vals & y=histogram height makes a nice swarm scatterplot that resembles a histogram
                if first_run[current_species_code]:
                    plot.plot(biovolume_vec, y, pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40), name=species_name)
                    first_run[current_species_code] = False
                else:
                    # to keep the legend from growing
                    plot.plot(biovolume_vec, y, pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40),)
        plot.setTitle('%s : year : %s' % (title, year))

    years_to_run_list = driver['simulation_years_logged']
    callback_obj = AnimationCallback(years_to_run_list, update)
    timer_calls.append(callback_obj)

def display_results_biovolume(min_dbh=0.0, mask=None, **kwargs):  #default: if no min_dbh is specified, all trees will be included; is no mask is specified, output from all plots will be shown
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_biovolume(h5file, driver, min_dbh, mask)

    def sum_by_year(mat):
        return np.sum(mat, axis=0)

    display_results_vs_time(title='Simulated Biovolume  (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='Biovolume (m^3/ha)',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_AGE_BIOVOLUME',
                            total_fn=sum_by_year,
                            total_name='Sum All')



def display_results_biomass(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_biomass(h5file, driver, min_dbh, mask)

    def sum_by_year(mat):
        return np.sum(mat, axis=0)

    display_results_vs_time(title='Simulated Biomass  (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='Biomass (Mg/ha)',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_AGE_BIOMASS',
                            total_fn=sum_by_year,
                            total_name='Sum All')


def display_results_leaf_area(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_leaf_area(h5file, driver, min_dbh, mask)

    display_results_vs_time(title='Simulated Leaf Area  (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='Leaf Area (m^2/ha)',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,)

# TODO: foliar biomass

def display_results_basal_area(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_basal_area(h5file, driver, min_dbh, mask)

    def sum_by_year(mat):
        return np.sum(mat, axis=0)

    display_results_vs_time(title='Simulated Basal Area (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='Basal Area (m^2/ha)',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_AGE_BASAL_AREA',
                            total_fn=sum_by_year,
                            total_name='Sum All')


def display_results_stems(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_stems(h5file, driver, min_dbh, mask)

    display_results_vs_time(title='Simulated Number of Stems (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='Stems/ha',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_STEMS')


def display_results_average_dbh(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_average_dbh(h5file, driver, min_dbh, mask)

    def max_by_year(mat):
        return np.max(mat, axis=0)

    display_results_vs_time(title='Average DBH (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='DBH/stem (cm/tree)',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_AGE_DBH',
                            total_fn=None, #max_by_year,
                            total_name='Max All')


def display_results_average_height(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_average_height(h5file, driver, min_dbh, mask)

    def max_by_year(mat):
        return np.max(mat, axis=0)

    display_results_vs_time(title='Average Height (min_dbh=%.1f)' % min_dbh,
                            bottom_label='Year',
                            left_label='Height (m/tree)',
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_HEIGHT',
                            total_fn=None, #max_by_year,
                            total_name='Max All')

def display_results_loreys_height(min_dbh=0.0, mask=None, **kwargs):
    years_in_sim_lst, \
    year_agg_mat, \
    num_species = compute_results_loreys_height(h5file, driver, min_dbh, mask)

    display_results_vs_time(title="Lorey's Height (min_dbh=%.1f)" % min_dbh,
                            bottom_label='Year',
                            left_label="Lorey's Height (m/tree)",
                            years_in_sim_lst=years_in_sim_lst,
                            year_agg_mat=year_agg_mat,
                            num_species=num_species,
                            expected_values_key='EXPECTED_HEIGHT',
                            total_fn=None,
                            total_name='Max All')


def display_results_vs_time(title,
                            bottom_label,
                            left_label,
                            years_in_sim_lst,
                            year_agg_mat,
                            num_species,
                            expected_values_key=None,
                            total_fn=None,
                            total_name='Total'):
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    legend = plot.addLegend()
    colors = make_colors(num_species+1)

    for current_species_code in range(num_species):
        species_name = driver['species_code_to_name'][current_species_code]
        # filter out nan values from the data
        full_years_vec = np.array(years_in_sim_lst)
        full_data_vec = year_agg_mat[current_species_code]
        not_nan_mask = ~np.isnan(full_data_vec)
        ydata_vec = full_data_vec[not_nan_mask]
        xdata_vec = full_years_vec[not_nan_mask]
        # plot this species' value
        if np.any(ydata_vec):
            curve = plot.plot(xdata_vec, ydata_vec, pen=colors[current_species_code], name=species_name)
        if expected_values_key and driver['species'][species_name].has_key(expected_values_key):
            xs, ys = driver['species'][species_name][expected_values_key]
            expected_curve = plot.plot(xs, ys, pen=None, symbol='+', symbolPen=colors[current_species_code], symbolBrush=None, symbolSize=8)

    # plot the total values
    if total_fn:
        year_agg_mat[np.isnan(year_agg_mat)] = 0.0
        total_vec = total_fn(year_agg_mat)
        pen = pg.mkPen(color=colors[-1], style=QtCore.Qt.CustomDashLine, width=3)
        # make a wide dashed line
        space = 8
        pen.setDashPattern([2,space])
        plot.plot(years_in_sim_lst, total_vec, pen=pen, name=total_name)  # QtCore.Qt.DotLine QtCore.Qt.DashLine

    # plot the expected totals
    if expected_values_key and driver.has_key(expected_values_key):
        xs, ys = driver[expected_values_key]
        expected_curve = plot.plot(xs, ys, pen=None, symbol='+', symbolPen=colors[-1], symbolBrush=None, symbolSize=10)

    plot.show()


def display_weather_dry_days(mask=None, **kwargs):
    display_weather_vs_time(title='Relative Dry Days Over Time',
                            bottom_label='Year',
                            left_label='Relative Dry Days',
                            group_name='RelativeDryDays',
                            mask=mask)

def display_weather_degree_days(mask=None, **kwargs):   #one value per plot
    display_weather_vs_time(title='Degree Days Over Time',
                            bottom_label='Year',
                            left_label='Degree Days',
                            group_name='DegreeDays',
                            mask=mask)

def display_weather_vs_time(title,
                            bottom_label,
                            left_label,
                            group_name,
                            mask):
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)

    years_in_sim_lst = driver['simulation_years_logged']
    num_years = len(years_in_sim_lst)
    weather_vec = np.zeros(num_years)
    for index, year in enumerate(years_in_sim_lst):
        year_str = '%.4d' % year
        # pull the current year matrix from the hdf file
        stored_matrix = np.array(h5file[group_name][year_str])
        avg_val = np.mean(stored_matrix[mask]) #this works even with None! Go Numpy!
        weather_vec[index] = avg_val

    curve = plot.plot(years_in_sim_lst, weather_vec)
    plot.show()
    print 'Simulation Average over Time = %s' %(np.mean(weather_vec))

def display_factor_growing_degree_days(mask=None, **kwargs):
    display_factor_type1(title='Average Growing Degree Day Factor',
                         bottom_label='Year',
                         left_label='Degree Days Factor (0 to 1)',
                         group_name='GrowingDegreeDaysFactor',
                         mask=mask)

def display_factor_soil_moisture(mask=None, **kwargs):
    display_factor_type1(title='Average Soil Moisture Factor',
                         bottom_label='Year',
                         left_label='Soil Moisture Factor (0 to 1)',
                         group_name='SoilMoistureFactor',
                         mask=mask)

def display_factor_soil_fertility(mask=None, **kwargs):
    display_factor_type1(title='Average Soil Fertility Factor',
                         bottom_label='Year',
                         left_label='Soil Fertility Factor (0 to 1)',
                         group_name='SoilFertilityFactor',
                         mask=mask)


# type 1 factor expect nx,ny,nspp matrices stored in the hdf file
def display_factor_type1(title,
                         bottom_label,
                         left_label,
                         group_name,
                         mask):
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    legend = plot.addLegend()
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species+1)]
    colors = make_colors(num_species)

    years_in_sim_lst = driver['simulation_years_logged']
    num_years = len(years_in_sim_lst)
    year_avgs_mat = np.zeros((num_species, num_years))

    for index, year in enumerate(years_in_sim_lst):
        year_str = '%.4d' % year
        # pull the current year factor matrix from the hdf file
        factor_matrix = np.array(h5file[group_name][year_str])  # size: nx,ny,nspp

        for current_species_code in range(num_species):
            species_name = driver['species_code_to_name'][current_species_code]
            # get all of the factor values for the species code of interest
            ##factor_mat = factor_matrix[:,:,current_species_code]
            factor_mat = factor_matrix[mask][:,current_species_code]
            # compute the average factor for this species on this year
            year_avg = np.mean( factor_mat )
            # store the avg for this species and this year
            year_avgs_mat[current_species_code, index] = year_avg

    for current_species_code in range(num_species):
        species_name = driver['species_code_to_name'][current_species_code]
        # plot this species' value
        # pull any nans out of the data
        data_vec = year_avgs_mat[current_species_code]
        nanmask = ~np.isnan(data_vec)
        ydata_vec = data_vec[nanmask]
        xdata_vec = np.array(years_in_sim_lst)[nanmask]
        curve = plot.plot(xdata_vec, ydata_vec, pen=colors[current_species_code], name=species_name)

    plot.show()


def display_factor_available_light(mask=None, **kwargs):
    ## height < 6m
    class1_min_height = 0.
    class1_max_height = 6.
    display_factor_type2(title='Average Available Light Factor : Height < %.0f m' % class1_max_height,
                         bottom_label='year',
                         left_label='AL Factor (0 to 1)',
                         group_name='AvailableLightFactor',
                         filter_by_height=True,
                         min_height=class1_min_height,
                         max_height=class1_max_height,
                         mask=mask)
    ## 6m >= height < 18m
    class2_min_height = 6.
    class2_max_height = 18.
    display_factor_type2(title='Average Available Light Factor : %.0f m <= Height < %.0f m' % (class2_min_height, class2_max_height),
                         bottom_label='year',
                         left_label='AL Factor (0 to 1)',
                         group_name='AvailableLightFactor',
                         filter_by_height=True,
                         min_height=class2_min_height,
                         max_height=class2_max_height,
                         mask=mask)
    ## height > 18m
    class3_min_height = 18.
    class3_max_height = 1e6
    display_factor_type2(title='Average Available Light Factor : Height > %.0f m' % class3_min_height,
                         bottom_label='year',
                         left_label='AL Factor (0 to 1)',
                         group_name='AvailableLightFactor',
                         filter_by_height=True,
                         min_height=class3_min_height,
                         max_height=class3_max_height,
                         mask=mask)
    
def display_factor_growth(mask=None, **kwargs):
    display_factor_type2(title='Average Growth Factor',
                         bottom_label='Year',
                         left_label='Growth Factor (0 to 1)',
                         group_name='GrowthFactor',
                         mask=mask)

# type 2 factor expect nx,ny,ntree matrices stored in the hdf file
def display_factor_type2(title,
                         bottom_label,
                         left_label,
                         group_name,
                         mask,
                         filter_by_height=False,
                         min_height=None,
                         max_height=None):
    print 'opening ', title
    # create the window that the swarm will be plotted to
    plot = pg.plot(title=title)   ## create an empty plot widget
    plot.setLabels(bottom=bottom_label, left=left_label)
    plot.setRange(yRange=(0,1))
    legend = plot.addLegend()
    num_species = len(driver['species_code_to_name'])
    colors = make_colors(num_species)

    years_in_sim_lst = driver['simulation_years_logged']
    num_years = len(years_in_sim_lst)
    year_avgs_mat = np.zeros((num_species, num_years)) + np.nan

    for index, year in enumerate(years_in_sim_lst):
        year_str = '%.4d' % year
        # pull the current year factor matrix from the hdf file
        factor_matrix = np.array(h5file[group_name][year_str])  # size: nx,ny,ntrees
        # pull the current year dbh matrix from the hdf file
        dbh_matrix = np.array(h5file['DBH'][year_str])
        # pull the current year species code matrix from the hdf file
        species_code_matrix = np.array(h5file['SpeciesCode'][year_str])
        if mask is not None:  #this is a hack
            species_code_matrix[np.logical_not(mask)] = -1  #set spp code for trees on plots that don't satisfy mask criteria to -1 (no tree); tree-level values are computed only w/in masked area

        for current_species_code in range(num_species):
            if filter_by_height:
                # get all of the dbh values for the species code of interest
                dbh_vec = dbh_matrix[species_code_matrix == current_species_code]
                # convert dbh into height
                species_name = driver['species_code_to_name'][current_species_code]
                fn = driver['species'][species_name]['TREE_HEIGHT_EQUATION']
                height_vec = fn(dbh_vec)
                # get all of the factor values for the species code of interest (same order as dbh_vec and height_vec)
                factor_vec = factor_matrix[species_code_matrix == current_species_code]
                selected_factor_vec = factor_vec[(height_vec >= min_height) & (height_vec < max_height)]
            else:
                selected_factor_vec = factor_matrix[species_code_matrix == current_species_code]

            if selected_factor_vec.size > 0:
                # compute the average factor for this species on this year
                year_avg = np.mean( selected_factor_vec )
                # store the avg for this species and this year
                year_avgs_mat[current_species_code, index] = year_avg

    if np.all(np.isnan(year_avgs_mat)):
        # show a little text message indicating that no trees of this size class were found
        ## Create text object, use HTML tags to specify color/size
        text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">No trees found in this size class.</span></div>',
                           anchor=(-0.3,1.3), border='w', fill=(0, 0, 255, 100))
        plot.addItem(text)
        text.setPos(0.1, 0.5)
    else:
        for current_species_code in range(num_species):
            species_name = driver['species_code_to_name'][current_species_code]
            if not np.all(np.isnan(year_avgs_mat[current_species_code])):
                # plot this species' value
                # pull any nans out of the data
                data_vec = year_avgs_mat[current_species_code]
                nanmask = ~np.isnan(data_vec)
                ydata_vec = data_vec[nanmask]
                xdata_vec = np.array(years_in_sim_lst)[nanmask]
                curve = plot.plot(xdata_vec, ydata_vec, pen=colors[current_species_code], name=species_name)

    plot.show()


class my_GLBarGraphItem(pggl.GLMeshItem):
    def __init__(self, pos, size, color):
        """
        pos is (...,3) array of the bar positions (the corner of each bar)
        size is (...,3) array of the sizes of each bar
        color is (nCubes, 4) array of rgba values (one for each bar)
        """
        nCubes = reduce(lambda a,b: a*b, pos.shape[:-1])
        cubeVerts = np.mgrid[0:2,0:2,0:2].reshape(3,8).transpose().reshape(1,8,3)
        cubeFaces = np.array([
            [0,1,2], [3,2,1],
            [4,5,6], [7,6,5],
            [0,1,4], [5,4,1],
            [2,3,6], [7,6,3],
            [0,2,4], [6,4,2],
            [1,3,5], [7,5,3]]).reshape(1,12,3)
        size = size.reshape((nCubes, 1, 3))
        pos = pos.reshape((nCubes, 1, 3))
        verts = cubeVerts * size + pos
        faces = cubeFaces + (np.arange(nCubes) * 8).reshape(nCubes,1,1)
        md = pggl.MeshData(verts.reshape(nCubes*8,3), faces.reshape(nCubes*12,3))

        cube_faces_colors = np.zeros((12*nCubes, 4))
        for i in range(nCubes):
            color_rgba = color[i]
            cube_faces_colors[i*12:(i+1)*12] = np.array([color_rgba] * 12)

        pggl.GLMeshItem.__init__(self, vertexes=verts.reshape(nCubes*8,3), faces=faces.reshape(nCubes*12,3), 
                                 faceColors=cube_faces_colors, shader='shaded', smooth=False)    

    def setData(self, pos, size, color):
        """
        Update the mesh data.
        pos is (...,3) array of the bar positions (the corner of each bar)
        size is (...,3) array of the sizes of each bar
        color is (nCubes, 4) array of rgba values (one for each bar)
        """        
        nCubes = reduce(lambda a,b: a*b, pos.shape[:-1])
        cubeVerts = np.mgrid[0:2,0:2,0:2].reshape(3,8).transpose().reshape(1,8,3)
        cubeFaces = np.array([
            [0,1,2], [3,2,1],
            [4,5,6], [7,6,5],
            [0,1,4], [5,4,1],
            [2,3,6], [7,6,3],
            [0,2,4], [6,4,2],
            [1,3,5], [7,5,3]]).reshape(1,12,3)
        size = size.reshape((nCubes, 1, 3))
        pos = pos.reshape((nCubes, 1, 3))
        verts = cubeVerts * size + pos
        faces = cubeFaces + (np.arange(nCubes) * 8).reshape(nCubes,1,1)
        md = pggl.MeshData(verts.reshape(nCubes*8,3), faces.reshape(nCubes*12,3))

        cube_faces_colors = np.zeros((12*nCubes, 4))
        for i in range(nCubes):
            color_rgba = color[i]
            cube_faces_colors[i*12:(i+1)*12] = np.array([color_rgba] * 12)

        self.setMeshData(vertexes=verts.reshape(nCubes*8,3), faces=faces.reshape(nCubes*12,3), 
                         faceColors=cube_faces_colors, shader='shaded', smooth=False)  

def display_spatial_max_height(title='Tree with maximum height in each plot'):
    global timer_calls
    print 'opening ', title
    win = pggl.GLViewWidget()
    #win.opts['distance'] = 130
    #win.setCameraPosition(distance=150)
    #win.setCameraPosition(distance=20, elevation=40., azimuth=5) #-90)   #this one works for 30x30 flat!
    win.setCameraPosition(distance=190, elevation=60., azimuth=90) #-90)    #this one works for 12x181 mountain!
    #print "Camera position = ", win.setCameraPosition.__doc__
    #print dir(win)
    #win.showFullScreen()
    win.pan(90,0,0)  #to center the mountain sim
    #assert(False)
    win.setWindowTitle(title)
    nx = driver['EW_number_of_plots']
    ny = driver['NS_number_of_plots']
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species+1)]
    colors = make_colors(num_species)

    #year = 250
    pos = np.zeros((nx*ny,3))
    size = np.zeros((nx*ny,3))
    color = np.zeros((nx*ny,4))
    bargraph = my_GLBarGraphItem(pos, size, color)
    win.addItem(bargraph)
    win.show()

    # the update function that will be called by the timer
    def update(year):
        if year==0:
            import time
            time.sleep(6)
        year_str = '%.4d' % year
        # pull the current year dbh matrix from the hdf file
        dbh_matrix = np.array(h5file['DBH'][year_str])
        # pull the current year species code matrix from the hdf file
        species_code_matrix = np.array(h5file['SpeciesCode'][year_str])

        n = 0
        for x in range(nx):
            for y in range(ny):
                tallest_tree = 0
                tallest_tree_species_code = 0
                # compute the species specific height values for each species
                for current_species_code in range(num_species):
                    # get all of the dbh values for the species code of interest
                    dbh_vec = dbh_matrix[x,y][species_code_matrix[x,y] == current_species_code]
                    if np.any(dbh_vec):
                        # convert dbh into height
                        species_name = driver['species_code_to_name'][current_species_code]
                        fn = driver['species'][species_name]['TREE_HEIGHT_EQUATION']
                        height_vec = fn(dbh_vec)
                        tallest_of_this_species = np.max(height_vec)
                        if tallest_of_this_species > tallest_tree:
                            tallest_tree = tallest_of_this_species
                            tallest_tree_species_code = current_species_code
                pos[n] = np.array([x,y,0])
                size[n] = np.array([0.4,0.4,tallest_tree])
                color[n] = np.array( [colors[tallest_tree_species_code].getRgbF()] )
                n += 1

        #bargraph = my_GLBarGraphItem(pos, size, color)
        #win.addItem(bargraph)
        bargraph.setData(pos, size, color)
        #win.show()
        win.setWindowTitle('%s : year : %s' % (title, year))


    years_to_run_list = driver['simulation_years_logged']
    callback_obj = AnimationCallback(years_to_run_list, update)
    timer_calls.append(callback_obj)

def display_scatter_factor_growing_degree_days(mask=None):
    return display_scatter_over_plot(plot_underlay_fn=display_equation_degree_day_factor,
                                     factor_group_name='GrowingDegreeDaysFactor',
                                     input_group_name='DegreeDays',
                                     mask=mask)

def display_scatter_factor_soil_moisture(mask=None):
    return display_scatter_over_plot(plot_underlay_fn=display_equation_soil_moisture_factor,
                                     factor_group_name='SoilMoistureFactor',
                                     input_group_name='RelativeDryDays',
                                     mask=mask)

def display_scatter_factor_soil_fertility(mask=None):
    return display_scatter_over_plot(plot_underlay_fn=display_equation_soil_fertility_factor,
                                     factor_group_name='SoilFertilityFactor',
                                     input_group_name='RelativeSoilFertility',
                                     mask=mask)

def display_scatter_over_plot(plot_underlay_fn,
                              factor_group_name,
                              input_group_name,
                              mask):
    global timer_calls
    print 'scatter : ',
    # plot the basic curve of the factor (x-axis is growing degree days, y-axis is computed factor)
    plot = plot_underlay_fn()
    # for each species plot the average factor for each plot on top of the equation curve
    num_species = len(driver['species_code_to_name'])
    #colors = [pg.intColor(i) for i in range(num_species+1)]
    colors = make_colors(num_species)
    scatter_plots = [plot.plot(pen=None, symbol='o', symbolSize=5, symbolPen=colors[current_species_code], symbolBrush=(255,255,255,40)) \
                      for current_species_code in range(num_species)]

    # the update function that will be called by the timer
    def update(year):
        year_str = '%.4d' % year
        # pull the current year factor matrix from the hdf file
        factor_matrix = np.array(h5file[factor_group_name][year_str])  # size: nx,ny,nspp
        # pull the growing degree days for this year
        ##growing_degree_days_vec = np.array(h5file[input_group_name][year_str]).reshape(-1)  #initially size:nx,ny, then reshaped to 1D vec nx*ny
        growing_degree_days_vec = np.array(h5file[input_group_name][year_str])[mask].reshape(-1)

        for current_species_code in range(num_species):
            species_name = driver['species_code_to_name'][current_species_code]
            # get all of the factor values for the species code of interest
            if mask is None:
                factor_vec = factor_matrix[:,:,current_species_code].reshape(-1)  #works
            else:
                factor_vec = factor_matrix[mask][:,current_species_code].reshape(-1)
            # scatter plot on top of the existing lines
            scatter_plots[current_species_code].setData(growing_degree_days_vec, factor_vec)
        plot.setTitle('year : %s' % (year))

    years_to_run_list = driver['simulation_years_logged']
    callback_obj = AnimationCallback(years_to_run_list, update)
    timer_calls.append(callback_obj)
    return plot


def display_DEM():
    print 'opening DEM window'
    data = np.random.normal(size=1000)
    win = pg.plot(data, title="Simplest possible plotting example")
    win.setLabels(bottom='dbh (cm)')


def main():
    import sys



    app = QtGui.QApplication([])

    ## setup the main window
    win = QtGui.QMainWindow()
    win.resize(800,700)
    win.setWindowTitle('Driver Information')
    cw = QtGui.QWidget()
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)
    win.setCentralWidget(cw)

    ### setup the python command console
    ## build an initial namespace for console commands to be executed in (this is optional;
    ## the user can always import these modules manually)
    namespace = {'pg': pg, 'np': np, 'driver':driver, 'h5file':h5file,
                 'display':display, 'swarm':swarm, 'equation':equation, 'results':results,
                 'weather':weather, 'factor':factor, 'spatial':spatial, 'scatter':scatter, 'make_raster':make_raster}

    command_info =  '\nCustom commands:\n'
    #command_info += "  display('DEM' | '?')\n"
    command_info += "  swarm('dbh' | 'bv' | 'rsf' | 'ht')\n"
    command_info += "  equation('ht' | 'bv' | 'bm' | 'la' | 'ba' | 'oi' | 'sff' | 'smf' | 'alf' | 'ddf' | 'all')\n"
    command_info += "  results('bv' | 'bm' | 'la' | 'ba' | 'stems' | 'dbh' | 'ht' | 'lht' | 'all', min_dbh=0, mask)\n"
    command_info += "  weather('gdd' | 'dry', mask)\n"
    command_info += "  factor('sff' | 'smf' | 'alf' | 'ddf' | 'growth' | 'all', mask)\n"
    command_info += "  spatial('ht' | 'all')\n"
    command_info += "  scatter('sff' | 'smf' | 'alf' | 'ddf' | 'all', mask)\n"
    command_info += "  make_raster('spp' | 'lht' | 'bm' | 'all', year, raster_filename)\n"   #spp = species code of the tree with the largest biomass on the plot. Need to specify year.

## TODO : 
#################
## grid displays
#################
# -- from driver --
# DEM (3D)
# fertility (pseudo color w/color bar)
# radiation on each plot
# -- from results --
# (DONE) height of tallest tree on each plot (3D) (w/crown shape ?)
# (DONE) species of tallest tree on each plot (pseudo color)
# species of larges biovolume on each plot (pseudo color)
# growing degree days on each plot - average across simulation (pseudo color)
# dry days on each plot - average across simulation (pseudo color)

#################
# 1D Y vs time plots
#################
# -- from results --
# (DONE) total biovolume/ha by species + total of all species
# (DONE) total biomass/ha by species + total of all species
# (DONE) total basal area/ha by species + total of all species
# (DONE) stems/ha by species + total of all species
# small, medium, and large stems/ha by species + total of all species
# (DONE) spatial average of dry days
# (DONE) spatial average of growing degree days
# (DONE) TODO store factors for each tree in the hdf file (maybe debug switch)
# (DONE) spatial average of species specific degree day factor
# (DONE) spatial average of species specific soil moisture factor
# (DONE) spatial average of species specific soil fertility factor
# (DONE) broken into 3 height classes:
# (DONE)    spatial average of species specific available light factor
# ground light factor
# inseeding factor
# total factor (by height class?)
# (DONE) average height vs time for each species
# average optimal dbh increment vs time for each species
# average actual dbh increment vs time for each species



    console = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=info + command_info)
    #console.show()
    console.resize(400,400)
    console.setWindowTitle('pyqtgraph example: ConsoleWidget')
    layout.addWidget(console, 0, 0)

    ## setup the display viewer
    #display_win = pg.GraphicsWindow()
    ##win.resize(800,700)
    #display_win.setWindowTitle('Swarm Histogram')
    #layout.addWidget(display_win, 0, 1)

    win.show()

    # start the gui running and wait to exit
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_file", help="specify the hdf file and path (if not in same directory)")
    args = parser.parse_args()
    hdf_file = args.hdf_file

    driver, h5file, info = load_driver(hdf_filename=hdf_file, vectorize='numba')  #'numpy' also works here
    # optimize the species specific function and turn them into numba ufuncs
    #driver = add_species_specific_ufuncs(driver)
    main()


