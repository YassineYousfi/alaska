## $Id$
## -*- coding: utf-8 -*-

# jpeg.dct
# ========
#   
# .. module:: pysteg.jpeg.dct
#   
# :Module:    pysteg.jpeg.dct
# :Date:      $Date$
# :Revision:  $Revision$
# :Copyright: © 2010: University of Surrey, UK
# :Author:    Hans Georg Schaathun <H.Schaathun@surrey.ac.uk> (2010)
# 
# ::

from numpy import dot,linalg
import numpy

def auxcos(x,u):
  return numpy.cos( (numpy.pi/8) * (x + 0.5) * u )

def cosmat(M=8,N=8):
  C = numpy.array( [ [ auxcos(x,u) for u in range(N) ] 
               for x in range(M) ] ) / 2
  C[:,0] = C[:,0] / numpy.sqrt(2)
  # C[0,:] = C[0,:] / numpy.sqrt(2)
  return C

auxM = cosmat(8,8)
invM = linalg.inv(auxM)
auxT = numpy.transpose(auxM)
invT = numpy.transpose(invM)

def dct2(g):
  """
    Perform a 2D DCT transform on g, assuming that g is 8x8.
  """
  assert (8,8) == numpy.shape( g )
  return dot( auxT, dot( g, auxM ) )

def idct2(g):
  """
    Perform a 2D inverse DCT transform on g, assuming that g is 8x8.
  """
  assert (8,8) == numpy.shape( g )
  # return dot( invM, dot( g, invT ) )
  return dot( invT, dot( g, invM ) )

def bdct(C,f=dct2):
  """
    Make a blockwise (8x8 blocks) 2D DCT transform on the matrix C.
    The optional second parameter f specifies the DCT transform function.
    The height and width of C have to be divisible by 8.
  """
  (M,N) = numpy.shape(C)
  assert M%8 == 0
  assert N%8 == 0
  S = numpy.ndarray((M,N))
  for i in range(0,M,8):
    for j in range(0,N,8):
      S[i:(i+8),j:(j+8)] = f( C[i:(i+8),j:(j+8)] )
  return S
      
def ibdct(C): return bdct(C,f=idct2)
