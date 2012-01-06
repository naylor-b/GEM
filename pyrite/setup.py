
from setuptools import setup
from setuptools.extension import Extension
from pkg_resources import get_build_platform

import os
import platform
import sys
    
# include search directories
incdirs = [
    'include',
]

# library search directories
libdirs = [
]

# libraries to link to
libs = [
]

# (-D) defines for the compiler
macros = [
    #('MY_DEFINE', '1'),
]

# other files to include in the distribution
data_files = [
]

try:
    import numpy
except ImportError:
    print 'numpy was not found.  Aborting build'
    sys.exit(-1)
else:
    incdirs.append(os.path.join(os.path.dirname(numpy.__file__), 'core', 'include'))

module1 = Extension('gem',
                    define_macros = macros,
                    include_dirs = incdirs,
                    libraries = libs,
                    library_dirs = libdirs,
                    language = 'c',
                    extra_link_args = [],
                    extra_compile_args = [],
                    sources = ['pygemmodule.c', 'gem.c'])


setup (name = 'pyrite',
       version = '0.1',
       description = 'A python wrapper for the GEM API',
       license = "Apache Version 2",
       py_modules = ['pyrite'],
       zip_safe = False,
       test_suite = 'test_pyrite',
       ext_modules = [module1],
       data_files = data_files)



