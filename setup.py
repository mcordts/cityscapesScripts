#!/usr/bin/python
#
# Enable cython support for eval scripts
# Run as
# setup.py build_ext --inplace
#
# For MacOS X you may have to export the numpy headers in CFLAGS
# export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

import os
from setuptools import setup, find_packages
has_cython = True

try:
    from Cython.Build import cythonize
except:
    print("Unable to find dependency cython. Please install for great speed improvements when evaluating.")
    print("sudo pip install cython")
    has_cython = False

include_dirs = []
try:
    import numpy as np
    include_dirs = np.get_include()
except:
    print("Unable to find numpy, please install.")
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
    'license': 'https://github.com/mcordts/cityscapesScripts/blob/master/license.txt',
    'version': '1.0.0',
    'install_requires': ['numpy', 'matplotlib', 'cython', 'pillow'],
    'packages': find_packages(),
    'scripts': [],
    'entry_points': {'gui_scripts': ['csViewer = cityscapesscripts.viewer.cityscapesViewer:main',
                                     'csLabelTool = cityscapesscripts.annotation.cityscapesLabelTool:main'],
                     'console_scripts': ['csEvalPixelLevelSemanticLabeling = cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling:main',
                                         'csEvalInstanceLevelSemanticLabeling = cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling:main',
                                         'csCreateTrainIdLabelImgs = cityscapesscripts.preparation.createTrainIdLabelImgs:main',
                                         'csCreateTrainIdInstanceImgs = cityscapesscripts.preparation.createTrainIdInstanceImgs:main']},
    'package_data': {'': ['icons/*.png']},
    'ext_modules': ext_modules,
    'include_dirs': [include_dirs]
}

setup(**config)
