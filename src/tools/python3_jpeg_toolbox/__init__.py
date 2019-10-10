## $Id: __init__.py 1601 2010-07-18 19:35:43Z css1hs $
## -*- coding: utf-8 -*-

"""
Processing in the JPEG domain.
------------------------------

:Module:    pysteg.jpeg
:Date:      $Date: 2010-07-18 20:35:43 +0100 (Sun, 18 Jul 2010) $
:Revision:  $Revision: 1601 $
:Copyright: © 2010: University of Surrey, UK
:Author:    Hans Georg Schaathun <H.Schaathun@surrey.ac.uk> (2010)

The main purpose of this package is to get direct access to the
JPEG data in a compressed file, without decompressing.  
This functionality is provided by the :class:`jpeg` class,
which is the only member intended for export.

The core functionality is implemented in C, as the :class:`jpegObject` class,
following the pattern of Phil Sallee's toolbox for Matlab.  
This class is not intended for direct use.  The derived :class:`jpeg`
class gives additional functionality implemented in Python.
The :class:`jpegObject` class should never be used in itself.

The intention is to provide full support for compression and 
decompression as well, but this has not yet been implemented and
tested.

This package is implemented partly in C, and the C code is not
properly documented.  The main components are:

* jpeglib is the Independent JPEG Groups API for JPEG compression
* jpegObject defines a Python class used as a base class below.

"""

print("[pysteg.jpeg] $Id: __init__.py 2204 2011-04-05 11:43:38Z georg $")

from .jpegObject import jpegObject
__all__ = [ "jpeg" ]

# We need standard components from :mod:`numpy`, and some auxiliary
# functions from submodules.
#
# ::

import numpy.random as rnd
from numpy import shape
import numpy as np

from . import base
from .dct import bdct, ibdct

# The colour codes are defined in the JPEG standard.  We store
# them here for easy reference by name::

colorCode = {
  "GRAYSCALE" 	: 1,
  "RGB" 	: 2,
  "YCbCr"       : 3,
  "CMYK"        : 4,
  "YCCK"        : 5
}

# The JPEG class
# ==============
#
# A derived class with methods to extract and reinsert a sample

class jpeg(jpegObject):
  """
    The jpeg (derived from jpegObject) allows the user to extract
    a sequence of pseudo-randomly ordered jpeg coefficients for
    watermarking/steganography, and reinsert them.
  """

  def __init__(self,file=None,key=None,rndkey=True,image=None,
                    verbosity=1,**kw):
    """
      The constructor will return a new Object with data from the given file.

      The key is used to determine the order of the jpeg coefficients.
      If no key is given, a random key is extracted using 
      random.SystemRandom().
    """
    if image != None:
      raise(NotImplementedError, "Compression is not yet implemented")
    jpegObject.__init__(self,file,**kw)
    self.verbosity = verbosity
    if verbosity > 0:
      print("[jpeg.__init__] Image size %ix%i" % (self.coef_arrays[0].shape))
    if key != None:
      self.key = key
    elif rndkey: 
      self.key = [ base.sysrnd.getrandbits(16) for x in range(16) ]
    else:
      self.key = None
  def getkey(self):
    """Return the key used to shuffle the coefficients."""
    return self.key

# 1D Signal Representations
# -------------------------

  def rawsignal(self,mask=base.acMaskBlock):
    """
      Return a 1D array of AC coefficients.
      (Most applications should use getsignal() rather than rawsignal().)
    """
    R = []
    for X in self.coef_arrays:
      (h,w) = X.shape
      A = base.acMask(h,w,mask)
      R = np.hstack ( [ R, X[A] ] )
    return R
  def getsignal(self,mask=base.acMaskBlock):
    """Return a 1D array of AC coefficients in random order."""
    R = self.rawsignal(mask)
    if self.key == None:
      return R
    else:
      rnd.seed(self.key)
      return R[rnd.permutation(len(R))]

  def setsignal(self,R0,mask=base.acMaskBlock):
    """Reinserts AC coefficients from getitem in the correct positions."""
    if self.key != None:
      rnd.seed(self.key)
      fst = 0
      P = rnd.permutation(len(R0))
      R = np.array(R0)
      R[P] = R0
    else:
      R = R0
    for X in self.coef_arrays:
      s = X.size * 63/64
      (h,w) = X.shape
      X[base.acMask(h,w,mask)] = R[fst:(fst+s)]
      fst += s
    assert len(R) == fst
    return ;

# Histogram and Image Statistics
# ------------------------------

  def abshist(self,mask=base.acMaskBlock,T=8):
    """
      Make a histogram of absolute values for a signal.
    """
    A = abs( self.rawsignal(mask) ).tolist()
    L = len(A)
    D = { }
    C = 0
    for i in range(T+1):
      D[i] = A.count(i)
      C += D[i]
    D["high"] = L - C
    D["total"] = L
    return D
  def hist(self,mask=base.acMaskBlock,T=8):
    """
      Make a histogram of the jpeg coefficients.
      The mask is a boolean 8x8 matrix indicating the
      frequencies to be included.  This defaults to the
      AC coefficients.
    """
    A = self.rawsignal(mask).tolist()
    E = [ -np.inf ] + [ i for i in range(-T,T+2) ] + [ np.inf ]
    return np.histogram( A, E ) 
  def nzcount(self,*a,**kw):
    """Number of non-zero AC coefficients.
    
      Arguments are passed to rawsignal(), so a non-default mask could
      be specified to get other coefficients than the 63 AC coefficients.
    """
    R = list(self.rawsignal(*a,**kw))
    return len(R) - R.count(0)

# Access to JPEG Image Data
# -------------------------

  def getCompID(self,channel):
    """
      Get the index of the given colour channel.
    """
    # How do we adress different channels?
    colourSpace = self.jpeg_color_space ;
    if colourSpace == colorCode["GRAYSCALE"]:
      if channel == "Y": return 0
      elif channel == None: return 0
      else:
        raise Exception("Invalid colour space designator")
    elif colourSpace == colorCode["YCbCr"]:
      if channel == "Y": return 0
      elif channel == "Cb": return 1
      elif channel == "Cr": return 2
      else:
        raise Exception("Invalid colour space designator")
    raise NotImplementedError("Only YCbCr and Grayscale are supported.")
  def getQMatrix(self,channel):
    """
      Return the quantisation matrix for the given colour channel.
    """
    cID = self.getCompID(channel)
    return self.quant_tables[self.comp_info[cID]["quant_tbl_no"]]
  def getCoefMatrix(self,channel="Y"):
    """
      This method returns the coefficient matrix for the given
      colour channel (as a matrix).
    """
    cID = self.getCompID(channel)
    return self.coef_arrays[cID]

# Decompression 
# -------------

  def getSpatial(self,channel="Y"):
    """
      This method returns one decompressed colour channel as a matrix.
      The appropriate JPEG coefficient matrix is dequantised 
      (using the quantisation tables held by the object) and
      inverse DCT transformed.
    """
    X = self.getCoefMatrix(channel)
    Q = self.getQMatrix(channel)
    (M,N) = shape(X)
    assert M % 8 == 0, "Image size not divisible by 8"
    assert N % 8 == 0, "Image size not divisible by 8"
    D = X * base.repmat( Q, (M/8, N/8) )
    S = ibdct(D)
    #assert max( abs(S).flatten() ) <=128, "Image colours out of range"
    return (S + 128 ).astype(np.uint8)

# Complete, general decompression is not yet implemented::

  def getimage(self):
    """
      Decompress the image and a PIL Image object.
    """

# Probably better to use a numpy image/array.
#
#     ::

    raise NotImplementedError("Decompression is not yet implemented")

# We miss the routines for upsampling and adjusting the size
#
#     ::

    L = len(self.coef_arrays)
    im = []
    for i in range(L):
      C = self.coef_arrays[i]
      if C != None:
        Q = self.quant_tables[self.comp_info[i]["quant_tbl_no"]]
        im.append( ibdct( dequantise( C, Q ) ) )
    return Image.fromarray(im)


# Calibration
# -----------

  def getCalibrated(self,channel="Y",mode="all"):
    """
      Return a calibrated coefficient matrix for the given channel.
      Channel may be "Y", "Cb", or "Cr" for YCbCr format.
      For Grayscale images, it may be None or "Y".
    """
    S = self.getSpatial(channel)
    (M,N) = shape(S)
    assert M % 8 == 0, "Image size not divisible by 8"
    assert N % 8 == 0, "Image size not divisible by 8"
    if mode == "col":
       S1 = S[:,4:(N-4)]
       cShape = ( M/8, N/8-1 )
    else:
      S1 = S[4:(M-4),4:(N-4)]
      cShape = ( (M-1)/8, (N-1)/8 )
    D = bdct(S1 - 128)
    X = D / base.repmat( self.getQMatrix(channel), cShape )
    return np.round(X)

  def calibrate(self,*a,**kw):
     assert len(self.coef_arrays) == 1
     self.coef_arrays[0] = self.getCalibrated(*a,**kw)

  def getCalSpatial(self,channel="Y"):
    """
      Return the decompressed, calibrated, grayscale image.
      A different colour channel can be selected with the channel
      argument.
    """

# We calibrate the image, obtaining a JPEG matrix.
#    ::

    C = self.getCalibrated(channel)

# The rest is straight forward JPEG decompression.
#    ::

    (M,N) = shape(C)
    cShape = (M/8,N/8)
    D = C * base.repmat( self.getQMatrix(channel), cShape )
    S = np.round( ibdct(D) + 128 )
    return S.astype(np.uint8)

# .. toctree::
#    :maxdepth: 2
#
#    base.py.txt
#    dct.py.txt
#    compress.py.txt

