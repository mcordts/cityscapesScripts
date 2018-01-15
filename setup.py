#!/usr/bin/python
#
# Enable cython support for eval scripts
# Run as
# setup.py build_ext --inplace
#
# For MacOS X you may have to export the numpy headers in CFLAGS
# export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

import os
from setuptools import setup
has_cython = True

try:
    from Cython.Build import cythonize
except:
    print("Unable to find dependency cython. Please use pip to install: cython to get great speed improvements when evaluating")
    print("sudo pip install cython")
    has_cython = False

include_dirs = []
try:
    import numpy as np
    include_dirs = np.get_include()
except:
    print("Unable to find cython dependency numpy. Please use pip to install: numpy to get great speed improvements when evaluating")
    print("sudo pip install numpy")

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

pyxFile = os.path.join("cityscapesscripts", "evaluation", "addToConfusionMatrix.pyx")

ext_modules = []
if has_cython:
    ext_modules = cythonize(pyxFile)

config = {
    'name': 'cityscapesscripts',
    'description': 'The Cityscapes Dataset Scripts Repository',
    'author': 'Marius Cordts',
    'url': 'www.cityscapes-dataset.net',
    'download_url': 'www.cityscapes-dataset.net',
    'author_email': 'mail@cityscapes-dataset.net',
    'version': '0.1',
    'install_requires': ['numpy', 'matplotlib', 'cython', 'pillow'],
    'packages': ['cityscapesscripts', 'cityscapesscripts.viewer', 'cityscapesscripts.annotation', 'cityscapesscripts.evaluation', 'cityscapesscripts.helpers'],
    'scripts': [],
    'entry_points': {'gui_scripts': ['viewer = cityscapesscripts.viewer.cityscapesViewer:main',
                                     'labeltool = cityscapesscripts.annotation.cityscapesLabelTool:main']
                     },
    'package_data': {'': ['icons/*.png']},
    'ext_modules': ext_modules,
    'include_dirs': [include_dirs]
}

setup(**config)
