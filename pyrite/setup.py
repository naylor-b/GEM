
from setuptools import setup
from setuptools.extension import Extension
from pkg_resources import get_build_platform

import os
import os.path
import platform
import sys
from glob import glob
from subprocess import Popen, PIPE, STDOUT
import re

def fixpath(path):
    """normalize the name and expand env vars"""
    return os.path.normpath(os.path.expandvars(path))


def get_files(wildcard):
    """return normalized paths for wildcard after expanding env vars"""
    return glob(fixpath(wildcard))

top = os.environ['NPSS_TOP']
devtop = os.environ['NPSS_DEV_TOP']
plat = os.environ['NPSS_CONFIG']
npssPath = fixpath('$NPSS_TOP/bin/npss.'+plat)

# get the NPSS version
ver_reg = re.compile('"NPSS_[^"]*"')
f = open(os.path.join(top, 'Executive', 'src', 'Common', 'NPSSversion.H'))
fcontents = f.read()
m = ver_reg.search(fcontents).group()[1:-1]
strs = m.split('- Rev:')
npss_version = strs[0][5:].strip()+strs[1].strip()
    
incdirs = [
    os.path.join(devtop, 'Executive', 'include'),
    os.path.join(devtop, 'util', 'src', 'Generic'),
]

libdirs = [
    os.path.join(devtop, 'Executive', 'lib', plat),
    os.path.join(devtop, 'util', plat, 'lib'),
    os.path.join(devtop, 'Ports', 'lib', plat),
    os.path.join(devtop, 'ModelDelivery', 'lib', plat),
    os.path.join(top, 'lib', plat)
]

macros = [
    ('USE_NEW_STREAMS', '1'),
    ('USE_DLL', '1'), 
]


try:
    import numpy
except ImportError:
    print 'numpy was not found, so references to numpy will be removed from this module'
# ??? leave NUMPY out until we support it better
else:
    incdirs.append(os.path.join(os.path.dirname(numpy.__file__), 'core', 'include'))
    macros.append(('USE_NUMPY', '1'))

comp_dlms = []
comp_interps = []
metadata = []

if plat == 'nt':
    dlm_ext = 'dll'
    libdirs.append(os.path.join(top, 'bin'))
    libdirs.append(os.path.join(top, 'lib', plat))
    libs = ['ModelDelivery', 'newmatlib', 'AntlrLib', 'NPSS_LIB']
    exlinkargs = []
    #VC8 stuff
    #libdirs.append('F:/Python-2.5.1/PCbuild8/Win32')
    #os.environ["MSSdk"]=r'D:\Program Files\Microsoft Visual Studio 8\SDK\v2.0'
    #os.environ["DISTUTILS_USE_SDK"] = "1"
          
    exInstLibs = []
    for d in os.environ['LIB'].split(';'):
       if 'fortran\\Lib\\ia32' in d:
          exInstLibs.append(os.path.join(d,'libifcoremd.dll'))
          exInstLibs.append(os.path.join(d,'libmmd.dll'))
    if len(exInstLibs) == 0:
       print "can't find Intel FORTRAN libs"
       sys.exit(1)
    exInstLibs.append(os.path.join(top, 'bin', 'NPSS_LIB.dll'))
else:
    dlm_ext = 'sc'

    # Determine which Fortran library to use.
    status = os.system('g77 -v 1>/dev/null 2>/dev/null')
    if status == 0:
        fortran_lib = 'g2c'
    else:
        status = os.system('gfortran -v 1>/dev/null 2>/dev/null')
        if status == 0:
            fortran_lib = 'gfortran'
        else:
            print 'Unknown fortran library (no g77 and no gfortran)'
            sys.exit(1)
    libs = ['NPSS_LIB', 'NPSSgeneric', 'ModelDelivery',
            'newmat', 'antlr', 'readline', 'curses', fortran_lib, 'm', 'dl']

    exlinkargs = ['-fPIC', '-Wl,-E']
    if platform.system() == 'Linux':
        eggdirname = 'pyNPSS-'+npss_version+'-py'+sys.version[0:3]+'-'+get_build_platform()+'.egg'
        # ld on trigger (2.15.92.0.2 20040927) doesn't handle $ORIGIN.
        # ld on torpedo (2.17.50.0.6-6.el5 20061020) does.
        proc = Popen('ld -v', shell=True, stdout=PIPE, stderr=STDOUT)
        (out, err) = proc.communicate()
        print "ld -v output '%s'" % out.rstrip()
        start = out.find(' 2.')
        if start > 0:
            ld_ver = float(out[start+1:start+5])
            print 'ld_ver', ld_ver
            if ld_ver >= 2.17:
                exlinkargs.extend(['-Wl,-z origin',
                                   '-Wl,-R ${ORIGIN}:${ORIGIN}/../'+eggdirname])
            else:
                print "This version of ld doesn't support $ORIGIN"
                print "You'll have to rely on LD_LIBRARY_PATH to find libNPSS.so"
        else:
            print "Can't determine version of ld"
            print "You'll have to rely on LD_LIBRARY_PATH to find libNPSS.so"

    exInstLibs = [os.path.join(top, 'lib', plat, 'libNPSS_LIB.so')]


for cdir in ['AirBreathing', 'Rockets', 'ControlsTB', 'DataViewers',
             'Executive', 'ThermoPackages', 'WATE++']:
    comp_dlms += get_files('$NPSS_TOP/'+cdir+'/DLMComponents/'+plat+'/*.'+dlm_ext) 
    comp_interps += get_files('$NPSS_TOP/'+cdir+'/InterpComponents/*.int') 
    metadata += get_files('$NPSS_TOP/'+cdir+'/MetaData/'+plat+'/*.met') 

metadata += get_files('$NPSS_TOP/MetaData/*.met')
metadata += get_files('$NPSS_TOP/WATE++/*.in')

data_files = [('', exInstLibs),
              ('InterpIncludes', get_files('$NPSS_TOP/InterpIncludes/*')),
              (os.path.join('DLMComponents', plat), comp_dlms),
              ('InterpComponents', comp_interps),
              (os.path.join('MetaData', plat), metadata),
              ('', ['test_npss.py', 'pax300.test', 'README'])]


module1 = Extension('pyNPSS',
                    define_macros = macros,
                    include_dirs = incdirs,
                    libraries = libs,
                    library_dirs = libdirs,
                    language = 'c++',
                    extra_link_args = exlinkargs,
                    extra_compile_args = ['-DUSE_DLL'],
                    sources = ['pyNPSS.cpp'])


# Taken from NPSScopyright.H
license = """\
Controlled Distribution.  Further distribution requires written approval of
the NASA Glenn Research Center, Cleveland, OH. or Wolverine Ventures, Inc.
under the provisions in the Space Act Agreement 3-621. Neither title nor
ownership of the software is hereby transferred.

Copyright 1997, 2001, 2002.  The United States Government as represented by
the Administrator of the National Aeronautics and Space Administration.
All rights reserved.

DISCLAIMER.
This software is provided "as is" without any warranty of any kind, either
expressed, implied, or statutory, including, but not limited to, any warranty
that the software will conform to specifications, any implied warranties of
merchantability, fitness for a particular purpose, or freedom from
infringement, any warranty that the software will be error free, or any
warranty that documentation, if provided, will conform to the software.
In no event shall the U.S. Government, or the U.S. Government's contractors or
subcontractors, be liable for any damages, including, but not limited to,
direct, indirect, special or consequential damages, arising out of, resulting
from, or in any way connected with this software, whether or not based upon
warranty, contract, tort, or otherwise, whether or not injury was sustained
by persons or property of otherwise, and whether or not loss was sustained
from, or arose out of the results of, or use of, the software or services
provided hereunder.
"""


setup (name = 'pyNPSS',
       version = npss_version,
       description = 'A python wrapper for NPSS',
       license = license,
       py_modules = ['npss'],
       zip_safe = False,
       test_suite = 'test_npss',
       ext_modules = [module1],
       data_files = data_files)



