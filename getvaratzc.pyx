import cython
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] getvaratzc(np.ndarray[DTYPE_t, ndim=4] varin,
                                             np.ndarray[DTYPE_t] z,
                                             np.ndarray[DTYPE_t, ndim=4] e):
    """Cython function to get a variable at fixed depths"""
    assert varin.dtype == DTYPE
    assert z.dtype == DTYPE
    assert e.dtype == DTYPE
    
    cdef unsigned int nt, nl, ny, nx, nz
    cdef unsigned int i, j, k, l, m
    nt, nl, ny, nx = np.shape(varin)
    nz = np.size(z)
    cdef np.ndarray[DTYPE_t, ndim=4] varout = np.full((nt,nz,ny,nx), 
                                                      0, dtype=DTYPE)
    
    for l in range(nt):
        for k in range(nl):
            for j in range(ny):
                for i in range(nx):
                    for m in range(nz):
                        if (e[l,k,j,i] - z[m])*(e[l,k+1,j,i] - z[m]) <= 0:
                            varout[l,m,j,i] = varin[l,k,j,i]
    return varout

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=5] getvaratzc5(np.ndarray[DTYPE_t, ndim=5] varin,
                                              np.ndarray[DTYPE_t] z,
                                              np.ndarray[DTYPE_t, ndim=4] e):
    """Cython function to get a variable at fixed depths"""
    assert varin.dtype == DTYPE
    assert z.dtype == DTYPE
    assert e.dtype == DTYPE
    
    cdef unsigned int nt, nl, ny, nx, nz
    cdef unsigned int i, j, k, l, m, n
    nt, nl, ny, nx, nv = np.shape(varin)
    nz = np.size(z)
    cdef np.ndarray[DTYPE_t, ndim=5] varout = np.full((nt,nz,ny,nx,nv), 
                                                      0, dtype=DTYPE)
    
    for l in range(nt):
        for k in range(nl):
            for j in range(ny):
                for i in range(nx):
                    for n in range(nv):
                        for m in range(nz):
                            if (e[l,k,j,i] - z[m])*(e[l,k+1,j,i] - z[m]) <= 0:
                                varout[l,m,j,i,n] = varin[l,k,j,i,n]
    return varout


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] getTatzc(np.ndarray[DTYPE_t] zl,
                                           np.ndarray[DTYPE_t] z,
                                           np.ndarray[DTYPE_t, ndim=4] e):
    """Cython function to get a variable at fixed depths"""
    assert zl.dtype == DTYPE
    assert z.dtype == DTYPE
    assert e.dtype == DTYPE
    
    cdef unsigned int nt, nl, ny, nx, nz
    cdef unsigned int i, j, k, l, m
    nt, nl, ny, nx = np.shape(e)
    nl = np.size(zl)
    nz = np.size(z)
    cdef np.ndarray[DTYPE_t, ndim=4] varout = np.full((nt,nz,ny,nx), 
                                                      0, dtype=DTYPE)
    
    for l in range(nt):
        for k in range(nl):
            for j in range(ny):
                for i in range(nx):
                    for m in range(nz):
                        if (e[l,k,j,i] - z[m])*(e[l,k+1,j,i] - z[m]) <= 0:
                            varout[l,m,j,i] = zl[k]
    return varout

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] getTatzc2(np.ndarray[DTYPE_t] zi,
                                           np.ndarray[DTYPE_t] z,
                                           np.ndarray[DTYPE_t, ndim=4] e):
    """Cython function to get a variable at fixed depths"""
    assert zi.dtype == DTYPE
    assert z.dtype == DTYPE
    assert e.dtype == DTYPE
    
    cdef unsigned int nt, nl, ny, nx, nz
    cdef unsigned int i, j, k, l, m
    nt, nl, ny, nx = np.shape(e)
    nl = np.size(zi) - 1.0
    nz = np.size(z)
    cdef np.ndarray[DTYPE_t, ndim=4] varout = np.full((nt,nz,ny,nx), 
                                                      0, dtype=DTYPE)
    
    for l in range(nt):
        for k in range(nl):
            for j in range(ny):
                for i in range(nx):
                    for m in range(nz):
                        if (e[l,k,j,i] - z[m])*(e[l,k+1,j,i] - z[m]) <= 0:
                            varout[l,m,j,i] = (zi[k] +
                            ((zi[k+1]-zi[k])/(e[l,k+1,j,i]-e[l,k,j,i])*(z[m]-e[l,k,j,i])))
    return varout
