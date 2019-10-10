## $Id: __init__.py 1601 2010-07-18 19:35:43Z css1hs $
## -*- coding: utf-8 -*-

# jpeg.base
# =========
#   
# .. module:: pysteg.jpeg.base
#   
# :Module:    pysteg.jpeg.base
# :Date:      $Date: 2010-07-18 20:35:43 +0100 (Sun, 18 Jul 2010) $
# :Revision:  $Revision: 1601 $
# :Copyright: © 2010: University of Surrey, UK
# :Author:    Hans Georg Schaathun <H.Schaathun@surrey.ac.uk> (2010)
# 
# ::

from numpy import array, vstack, hstack, kron, ones

# The sysrnd variable is defined here for the benefit of other submodules::

from random import SystemRandom
sysrnd = SystemRandom()

# ::

def uMask(u=7): 
  return array([ [ (x+y<u) & (x+y>0) for x in range(8) ] for y in range(8) ])

# Make a mask to extract AC coefficient

acMaskBlock = array (
  [ [ False ] + [ True for x in range(7) ] ] +
  [ [ True for x in range(8) ] for y in range(7) ] 
)

def acMask(h1,w1,mask=acMaskBlock):
  """Return a mask of the given size for the AC coefficients of a
     JPEG coefficient matrix."""
  (h,w) = (h1/8,w1/8)
  A = vstack( [ mask for x in range(h) ] )
  A = hstack( [ A for x in range(w) ] )
  return A 

def repmat(M,rep): return kron( ones( rep ), M )

def getfreq(A,i,j):
  """
    Return a submatrix of a JPEG matrix A including only frequency (i,j)
    from each block.
  """
  (M,N) = A.shape
  return A[xrange(i,M,8),:][:,xrange(j,N,8)]

