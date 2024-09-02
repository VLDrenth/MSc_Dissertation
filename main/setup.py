# File: setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "kmeans_init_float",
        ["kmeans_init_float.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)