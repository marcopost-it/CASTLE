#import os
#
#import numpy
#from Cython.Build import cythonize
#
#def configuration(parent_package='', top_path=None):
#    
#    from numpy.distutils.misc_util import Configuration
#    
#    libraries = []
#    if os.name == 'posix':
#        libraries.append('m')
#
#    config = Configuration('CASTLE', parent_package, top_path)
#    config.add_extension('_puredbscan_inner',
#                         sources=['_puredbscan_inner.pyx'],
#                         include_dirs=[numpy.get_include()],
#                         language="c++")
#    config.ext_modules[-1] = cythonize(config.ext_modules[-1])
#
#    return config
#
#
#if __name__ == '__main__':
#    from numpy.distutils.core import setup
#    setup(**configuration(top_path='').todict())

import numpy
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("_puredbscan_inner",
                               ["_puredbscan_inner.pyx"],
                               language='c++',
                               include_dirs=[numpy.get_include()],)]
)

#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(name='_puredbscan_inner',
#      ext_modules=cythonize("clustering\\_puredbscan_inner.pyx"))