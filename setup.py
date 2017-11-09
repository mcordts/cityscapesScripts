#!/usr/bin/python


try:
    import numpy as np
except ImportError:
    print('install numpy first')
from setuptools import Extension, find_packages, setup

description = (
        'This repository contains scripts for'
        'inspection, preparation, and evaluation of the Cityscapes dataset.')
c_file = 'cityscapesscripts/evaluation/addToConfusionMatrix.c'
include_file = 'cityscapesscripts/evaluation/addToConfusionMatrix_impl.c'
extensions = [Extension(
        'cityscapesscripts.evaluation.addToConfusionMatrix',
        [c_file],
        include_dirs=[np.get_include(), include_file])]
setup(
        name='cityscapesScripts',
        version='1.0.4',
        description=description,
        url='https://github.com/mcordts/cityscapesScripts',
        author='mcordts',
        license='custon',

        packages=find_packages(),
        install_requires=['numpy', 'pillow', 'matplotlib'],
        setup_requires=['setuptools>=18.0', 'numpy'],
        ext_modules=extensions,
)
