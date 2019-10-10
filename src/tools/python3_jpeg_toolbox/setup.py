#! /usr/bin/env python
# $Id: setup.py 1494 2009-04-30 13:10:41Z css1hs $

from distutils.core import setup, Extension
import numpy.distutils.misc_util
setup(
  ext_modules = [
    Extension ( "jpegObject",
      sources=["jpegobject.c"],
      library_dirs=[ "jpeglib" ],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
      libraries=[ "jpeg" ],
    ),
  ],
)

# Note that the include_dirs given here are due to a bug in Ubuntu.
# -> Check it for your system.
