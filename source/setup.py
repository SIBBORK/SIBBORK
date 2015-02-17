#import numpy
#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(
#    ext_modules = cythonize("light3d.pyx", include_dirs=[numpy.get_include()])
#)


"""
Build the python extension of the TUSBTunerDriver C++ class.
This allows us to interface with the Tuner Board using python.
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_options = {
               "sources": ["light3d.pyx"],
               "include_dirs": ['cpp', numpy.get_include()]
               }

extmodule = Extension("light3d", **ext_options)
# Building
setup(
    name = 'light3d',
    version = '1.0',
    description = 'Cythonized Light3D routine.',
    author = "Author Here",
    long_description = 'This python package if a fater implementation of the 3D light routine.',
    ext_modules = [extmodule],
    cmdclass = {'build_ext': build_ext},
)                                  

