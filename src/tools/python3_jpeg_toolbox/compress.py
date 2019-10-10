## $Id: __init__.py 1601 2010-07-18 19:35:43Z css1hs $
## -*- coding: utf-8 -*-

# jpeg.compress
# =============
#   
# .. module:: pysteg.jpeg.compress
#   
# :Module:    pysteg.jpeg.compress
# :Date:      $Date: 2010-07-18 20:35:43 +0100 (Sun, 18 Jul 2010) $
# :Revision:  $Revision: 1601 $
# :Copyright: © 2010: University of Surrey, UK
# :Author:    Hans Georg Schaathun <H.Schaathun@surrey.ac.uk> (2010)
# 
# Importing everything from :mod:`pylab` is usually a bad idea.
# What do we really need?
# ::

from pylab import *
from base import repmat

# The standard quantisation tables for JPEG::

table0 = array(
          [ [ 16,  11,  10,  16,  24,  40,  51,  61 ],
	    [ 12,  12,  14,  19,  26,  58,  60,  55 ],
	    [ 14,  13,  16,  24,  40,  57,  69,  56 ],
	    [ 14,  17,  22,  29,  51,  87,  80,  62 ],
	    [ 18,  22,  37,  56,  68, 109, 103,  77 ],
	    [ 24,  35,  55,  64,  81, 104, 113,  92 ],
	    [ 49,  64,  78,  87, 103, 121, 120, 101 ],
	    [ 72,  92,  95,  98, 112, 100, 103,  99 ] ] )

table1 = array(
          [ [ 17,  18,  24,  47,  99,  99,  99,  99 ],
	    [ 18,  21,  26,  66,  99,  99,  99,  99 ],
	    [ 24,  26,  56,  99,  99,  99,  99,  99 ],
	    [ 47,  66,  99,  99,  99,  99,  99,  99 ],
	    [ 99,  99,  99,  99,  99,  99,  99,  99 ],
	    [ 99,  99,  99,  99,  99,  99,  99,  99 ],
	    [ 99,  99,  99,  99,  99,  99,  99,  99 ],
	    [ 99,  99,  99,  99,  99,  99,  99,  99 ] ] )

# The quantTable function seems straight forward,
# but has not been tested.

def quantTable(quality=50,tnum=0,force_baseline=False):
  if quality <= 0: quality = 1
  elif quality > 100: quality = 100
  if quality < 50: quality = 5000 / quality 
  else: quality = 200 - quality*2

  t = floor( (t * quality + 50) /100 )

  t[t<1] = 1 

  if (force_baseline): t[t>255] = 255
  else: t[t>32767] = 32767  # max quantizer needed for 12 bits

  return t

# I don't think this works.

def bdctmtx(n=8):
  (c,r) = meshgrid(range(n), range(n))
  (c0,r0) = meshgrid(r.flatten());
  (c1,r1) = meshgrid(c.flatten());

  x = sqrt(float(2) / n) 
  x *= cos( pi * (2*c + 1) * r / (2 * n));
  x[1,:] = x[1,:] / sqrt(2);

  return x[r0+c0*n+1] * x[r1+c1*n+1]

def im2vec(im,blksize=8,padsize=0):
  """Reshape 2D image blocks into an array of column vectors

     V=im2vec(im,blksize=8,padsize=0)
 
     IM is an image to be separated into non-overlapping blocks and
     reshaped into an MxN array containing N blocks reshaped into Mx1
     column vectors.  im2vec is designed to be the inverse of vec2im.
 
     BLKSIZE is a scalar or 1x2 vector indicating the size of the blocks.
 
     PADSIZE is a scalar or 1x2 vector indicating the amount of vertical
     and horizontal space to be skipped between blocks in the image.
     Default is [0 0].  If PADSIZE is a scalar, the same amount of space
     is used for both directions.  PADSIZE must be non-negative (blocks
     must be non-overlapping).
 
     ROWS indicates the number of rows of blocks found in the image.
     COLS indicates the number of columns of blocks found in the image.
  """
 
  blksize=blksize + array( [0,0] )
  padsize=padsize + array( [0,0] )
  if ( any( padsize < 0 ) ):
    raise InputException, "Pad size must be non-negative." 

  (height,width) = im.shape
  (y,x) = blksize + padsize

  rows = int( ( height + padsize[0] ) / y )
  cols = int( ( width + padsize[1] ) / x )

  T = zeros( [y*rows,x*cols] )

  imy = y*rows - padsize[0]
  imx = x*cols - padsize[1]

  T[0:imy,0:imx] = im[0:imy,0:imx]

  T = reshape(T, [ cols, y, rows, x ] )
  T = transpose( T, [0,2,1,3])
  T = reshape( T, [ y, x, rows*cols ] ) 
  V = T[0:blksize[0], 0:blksize[1], 0:(rows*cols) ]

  return (reshape( V, [ rows*cols, y*x ] ), rows, cols)

def vec2im(V,blksize=None,padsize=0,rows=None,cols=None):
  """Reshape and combine column vectors into a 2D image
 
     V is an MxN array containing N Mx1 column vectors which will be reshaped
     and combined to form image IM. 
 
     PADSIZE is a scalar or a 1x2 vector indicating the amount of vertical and
     horizontal space to be added as a border between the reshaped vectors.
     Default is [0 0].  If PADSIZE is a scalar, the same amount of space is used
     for both directions.
 
     BLKSIZE is a scalar or a 1x2 vector indicating the size of the blocks.
     Default is sqrt(M).
 
     ROWS indicates the number of rows of blocks in the image. Default is
     floor(sqrt(N)).
 
     COLS indicates the number of columns of blocks in the image.  Default
     is ceil(N/ROWS).
  """

  (n,m) = V.shape

  padsize = padsize + array( [0,0] )
  if ( any( padsize < 0 ) ):
    raise InputException, "Pad size must be non-negative." 

  if blksize == None:
    bsize = floor(sqrt(m))

  blksize = blksize + array([0,0])
  if prod(blksize) != m:
    print(m, blksize)
    raise InputException, 'Block size does not match size of input vectors.'

  if rows == None:
    rows = floor(sqrt(n))
  if cols == None:
    cols = ceil(n/rows)

# make image
#
#   ::

  (y,x) = blksize + padsize

# zero() returns float64 and causes T to become a floating point array
# This is a bug; integer input should give integer return
#
#   ::

  T = zeros( [rows*cols, y, x] )
  T[0:n, 0:blksize[0], 0:blksize[1]] = \
    reshape( V, [n, blksize[0], blksize[1] ] )
  T = reshape(T, [rows,cols,y,x] )
  T = transpose( T, [0,2,1,3] )
  T = reshape(T, [y*rows,x*cols] )
  return T[0:(y*rows-padsize[0]), 0:(x*cols-padsize[1]) ]

#def quantise(C,Q):
 # """Quantise DCT coefficients using the quantisation matrix Q."""
  #return C / repmat( Q, C.shape / Q.shape )
def quantise(C,Q):
  """Quantise DCT coefficients using the quantisation matrix Q."""
  [k,l] = C.shape
  [m,n] = Q.shape
  rep = (k/m, l/n)
  return C / repmat(Q, rep)

#def dequantise(C,Q):
 # """Dequantise JPEG coefficients using the quantisation matrix Q."""
  #return C * repmat( Q, C.shape / Q.shape )
def dequantise(C,Q):
  """Dequantise JPEG coefficients using the quantisation matrix Q."""
  [k,l] = C.shape
  [m,n] = Q.shape
  rep = (k/m, l/n)
  return C * repmat(Q, rep)
  
def bdct(A,blksize=8):
  """Blocked discrete cosine transform for JPEG compression."""
  dctm = bdctmtx(blksize)
  (v,r,c) = im2vec(a,blksize)
  return vec2im(dot(dctm,v),blksize,rows=r,cols=c)

def ibdct(A,blksize=8):
  """Inverse blocked discrete cosine transform"""
  dctm = bdctmtx(blksize)
  (v,r,c) = im2vec(a,blksize)
  return vec2im(dot(transpose(dctm),v),blksize,rows=r,cols=c)

