from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


ext_modules = [Extension("BGTools_cython", ["BGTools_cython.pyx"])]

setup(
  name = 'BGTools_cython',
  cmdclass = {'build_ext':build_ext},
  include_dirs = [numpy.get_include()],
  ext_modules = ext_modules,
)
