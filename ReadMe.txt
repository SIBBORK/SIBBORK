SIBBORK is an individual-based spatially-explicit gap dynamics model for simulation of forests, here specifically tailored to boreal forests in Siberia. If you use this code, please reference:

Brazhnik and Shugart (2015) SIBBORK: New Spatially-Explicit Gap Model for Boreal Forest in 3-Dimensional Terrain. Ecological Modelling, in review.

This model is licensed under the GNU General Public License v 2.0 (see LICENSE.txt).

Introductory video is available at: https://vimeo.com/139717413
The model is described in the file SIBBORK_ModelDevelopment.PDF

A full user's manual and tutorials will be uploaded later in 2015 (check back in the fall).

--

The model utilizes many open source libraries. In order to run the model, you will need the following:

python 2.7:
https://www.python.org/download/releases/2.7.6

updated numpy library:
https://pypi.python.org/pypi/numpy/1.8.1

h5py library:
https://pypi.python.org/pypi/h5py/2.3.0

GDAL library:
https://pypi.python.org/pypi/GDAL/1.10.0

cython:
https://pypi.python.org/pypi/Cython/0.20.1

numba:
https://numba.pydata.org

mako:
https://www.makotemplates.org

--

Install:
    clone this project from commandline via:
        git clone https://github.com/SIBBORK/SIBBORK.git
    cd into source folder
    compile the 3-D light ray tracing subroutine using cython from commandline via:
        python2 setup.py build_ext --inplace

Invoke the model from commandline via:

    python2 sibbork.py driver_OptIncCompare.py <folder>/<output_filename>.hdf5

The output file is in Hierarchichal Data Format (HDF) and can be viewed with vitables in Linux or HDFviewer on a Windows platform.
The output files can be large, on the order of 500MB, especially when running for 1000 years and multiple species. Make sure you have enough space, or make the file smaller by outputting less frequently or running for fewer years. State variables are output at the user-specified interval in the driver file (line 492). Run duration is specified in the driver (lines 432-433).

