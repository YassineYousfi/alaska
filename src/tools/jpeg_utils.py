import os
import numpy as np
from scipy import misc
import random
import scipy.io as sio
from scipy import fftpack
import sys
from os.path import expanduser
home = expanduser("~")
user = home.split('/')[-1]
sys.path.append(home + '/alaska_github/src/')
import tools.python3_jpeg_toolbox as jpeglib

from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt

def block_view(A, block= (8,8)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]// block[0], A.shape[1]// block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return as_strided(A, shape= shape, strides= strides)

def segmented_stride(M, fun, blk_size=(8,8), overlap=(0,0)):
    # This is some complex function of blk_size and M.shape
    B = block_view(M, block=blk_size)
    B[:,:,:,:] = fun(B)
    return M

def decompress(S):
    # Decompress DCT coefficients C using quantization table Q
    assert S.coef_arrays[0].shape[0] % 8 == 0, 'Wrong image size'
    assert S.coef_arrays[0].shape[1] % 8 == 0, 'Wrong image size'
    I = np.zeros_like(S.coef_arrays,dtype=np.float64)
    for i in range(I.shape[0]):
        Q = S.quant_tables[S.comp_info[i]['quant_tbl_no']]
        # this multiplication is done on integers
        fun = lambda x : np.multiply(x,Q)
        C = np.float64(segmented_stride(S.coef_arrays[i], fun)) 
        fun = lambda x: fftpack.idct(fftpack.idct(x, norm='ortho',axis=2), norm='ortho',axis=3) + 128
        I[i,:,:] = segmented_stride(C, fun)
    return np.expand_dims(np.swapaxes(I,0,2),0)

def branch_to_slice(branch):
    if branch == 'YCrCb':
        return slice(None)
    if branch == 'Y':
        return slice(1)
    if branch == 'CrCb':
        return slice(2)
    if branch == 'Cr':
        return slice(1,2)
    if branch == 'Cb':
        return slice(2,3)