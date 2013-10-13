cimport cython
from cython.view cimport array as cvarray
import numpy as np
import math
cdef extern from "mabonlm3d.h":
    void mabonlm3d_c(double *ima, int *dims, int v, int f, int r, double *fima)

cdef inline int _int_max(int a, int b): return a if a >= b else b
cdef inline int _int_min(int a, int b): return a if a <= b else b

def _firdn_vector(double[:] f, double[:] h, double[:] out):
    cdef int n=len(f)
    cdef int klen=len(h)
    cdef int outLen=(n+klen)//2
    cdef double ss
    cdef int i, k, limInf, limSup, x=0, ox=0, ks=0
    for i in range(outLen):
        ss=0
        limInf=_int_max(0, x-klen+1)
        limSup=1+_int_min(n-1, x)
        ks=limInf
        for k in range(limInf, limSup):
            ss+=f[ks]*h[x-k]
            ks+=1
        out[ox]=ss
        x+=2
        ox+=1

def _upfir_vector(double[:] f, double[:] h, double[:] out):
    cdef int n=f.shape[0]
    cdef int klen=h.shape[0]
    cdef int outLen=2*n+klen-2
    cdef int x, limInf, limSup, k, ks
    cdef double ss
    for x in range(outLen):
        limInf=_int_max(0, x-klen+1);
        if(limInf%2==1):
            limInf+=1
        limSup=_int_min(2*(n-1), x)
        if(limSup%2==1):
            limSup-=1
        ss=0
        k=limInf
        ks=limInf//2
        while(k<=limSup):
            ss+=f[ks]*h[x-k];
            k+=2;
            ks+=1
        out[x]=ss

def _firdn_matrix(double[:,:] F, double[:] h, double[:,:] out):
    cdef int n=F.shape[0]
    cdef int m=F.shape[1]
    cdef int j
    for j in range(m):
        _firdn_vector(F[:,j], h, out[:,j])

def _upfir_matrix(double[:,:] F, double[:] h, double[:,:] out):
    cdef int n=F.shape[0]
    cdef int m=F.shape[1]
    for j in range(m):
        _upfir_vector(F[:,j], h, out[:,j]);

cpdef firdn(double[:,:] image, double[:] h):
    '''
    Applies the filter given by the convolution kernel 'h' columnwise to 
    'image', then subsamples by 2. This is a special case of the matlab's
    'upfirdn' function, ported to python. Returns the filtered image.
    Parameters
    ----------
        image:  the input image to be filtered
        h:      the convolution kernel
    '''
    nrows=image.shape[0]
    ncols=image.shape[1]
    ll=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=((nrows+ll)//2, ncols))
    _firdn_matrix(image, h, filtered)
    return filtered

cpdef upfir(double[:,:] image, double[:] h):
    '''
    Upsamples the columns of the input image by 2, then applies the 
    convolution kernel 'h' (again, columnwise). This is a special case of the 
    matlab's 'upfirdn' function, ported to python. Returns the filtered image.
    Parameters
    ----------
        image:  the input image to be filtered
        h:      the convolution kernel
    '''
    nrows=image.shape[0]
    ncols=image.shape[1]
    ll=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=(2*nrows+ll-2, ncols))
    _upfir_matrix(image, h, filtered)
    return filtered

def aonlm(double [:,:,:]image, int v, int f, int r):
    cdef double[:,:,:] I=image.copy_fortran()
    cdef double[:,:,:] filtered=I.copy_fortran()
    cdef int[:] dims=cvarray((3,), itemsize=sizeof(int), format="i")
    dims[0]=I.shape[0]
    dims[1]=I.shape[1]
    dims[2]=I.shape[2]
    mabonlm3d_c(&I[0,0,0], &dims[0], v, f, r, &filtered[0,0,0])
    return filtered
