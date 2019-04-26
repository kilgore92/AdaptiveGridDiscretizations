import numpy as np

# Dot product (vector-vector, matrix-vector and matrix-matrix) in parallel
def dot_VV(v,w):
    if v.shape!=w.shape: raise ValueError('dot_VV : Incompatible shapes')
    return np.multiply(v,w).sum(0)

def dot_AV(a,v):
    m,n = a.shape[:2]
    bounds = a.shape[2:]
    if v.shape != (n,)+bounds:
        raise ValueError("dot_AV : Incompatible shapes")

    return np.multiply(a,\
        np.broadcast_to(np.reshape(v,(1,n,)+bounds), (m,n,)+bounds) \
        ).sum(1)

def dot_AA(a,b):
    m,n=a.shape[:2]
    bounds = a.shape[2:]
    k = b.shape[1]
    if b.shape!=(n,k,)+bounds:
        raise ValueError("dot_AA error : Incompatible shapes")
    return np.multiply(
        np.broadcast_to(np.reshape(a,(m,n,1,)+bounds),(m,n,k,)+bounds),
        np.broadcast_to(np.reshape(b,(1,n,k,)+bounds),(m,n,k,)+bounds)
    ).sum(1)

def dot_VAV(v,a,w):
    return dot_VV(v,dot_AV(a,w))
    
# Multiplication by scalar, of a vector or matrix
def mult(k,x):
    bounds = k.shape
    dim = x.ndim-k.ndim
    if x.shape[dim:]!=bounds:
        raise ValueError("mult error : incompatible shapes")
    return np.multiply(
        np.broadcast_to(np.reshape(k,(1,)*dim+bounds),x.shape),
        x)
    

def perp(v):
    if v.shape[0]!=2:
        raise ValueError("perp error : Incompatible dimension")        
    return np.array( (-v[1],v[0]) )
    
def cross(v,w):
    if v.shape[0]!=3 or v.shape!=w.shape:
        raise ValueError("perp error : Incompatible dimensions")
    return np.array( (v[1]*w[2]-v[2]*w[1], \
    v[2]*w[0]-v[0]*w[2], v[0]*w[1]-v[1]*w[0]) )
    
def outer(v,w):
    if v.shape != w.shape:
        raise ValueError("outer error : Incompatible dimensions")
    d=v.shape[0]
    bounds = v.shape[1:]
    return np.multiply(
        np.broadcast_to(np.reshape(v,(d,1,)+bounds),(d,d,)+bounds),
        np.broadcast_to(np.reshape(v,(1,d,)+bounds),(d,d,)+bounds)
    )

def outer_self(v):
    return outer(v,v)

def transpose(a):
    return a.transpose( (1,0,)+tuple(range(2,a.ndim)) )
    
def trace(a):
    dim = a.shape[0]
    if a.shape[1]!=dim:
        raise ValueError("trace error : incompatible dimensions")
    return a[(range(dim),range(dim))].sum(0)

# Low dimensional special cases

def det(a):
    dim = a.shape[0]
    if a.shape[1]!=dim:
        raise ValueError("inverse error : incompatible dimensions")
    if dim==1:
        return a[0,0]
    elif dim==2:
        return a[0,0]*a[1,1]-a[0,1]*a[1,0]
    elif dim==3:
        return a[0,0]*a[1,1]*a[2,2]+a[0,1]*a[1,2]*a[2,0]+a[0,2]*a[1,0]*a[2,1] \
        - a[0,2]*a[1,1]*a[2,0] - a[1,2]*a[2,1]*a[0,0]- a[2,2]*a[0,1]*a[1,0]
    else:
        raise ValueError("det error : unsupported dimension") 
    
def inverse(a):
    dim = a.shape[0]
    if a.shape[1]!=dim:
        raise ValueError("inverse error : incompatible dimensions")
    if dim==1:
        return 1./a[0,0]
    elif dim==2:
        d=det(a)
        return np.array( ((a[1,1]/d,-a[0,1]/d),(-a[1,0]/d,a[0,0]/d)) )
    elif dim==3:
        d=det(a)
        return np.array([[(
        a[(i+1)%3,(j+1)%3]*a[(i+2)%3,(j+2)%3]-
        a[(i+2)%3,(j+1)%3]*a[(i+1)%3,(j+2)%3]
        )/d
        for i in range(3)] for j in range(3)])
    else: 
        raise ValueError("inverse error : unsupported dimension")
    
        