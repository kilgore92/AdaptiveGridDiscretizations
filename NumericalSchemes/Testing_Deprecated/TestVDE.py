import numpy as np
from LinearP import *

# Call python implementation
import Selling

# Call c++ library
FileVDE_binary_dir = '/Users/mirebeau/bin/VoronoiDecompExport/FileVDE/Debug' 
import FileIO

# Extract the lower triangular part of a (symmetric) matrix, put in a vector
def Sym2V(a):
	m,n = a.shape[:2]
	if m!=n:
		raise ValueError("Sym2V error: matrix is not square")
	return np.array([a[i,j] for i in range(n) for j in range(i+1)])

def V2Sym(a):
	d = a.shape[0]
	m = int(np.floor(np.sqrt(2*d)))
	if d!=m*(m+1)//2:
		raise ValueError("V2Sym error: first dimension not of the form d(d+1)/2")
	def index(i,j):
		a,b=np.maximum(i,j),np.minimum(i,j)
		return a*(a+1)//2+b
	return np.array([ [a[index(i,j)] for i in range(m)] for j in range(m)])

def VoronoiDecomp(a):
	vdeIn ={'tensors':np.moveaxis(Sym2V(a),0,-1)}
	vdeOut = FileIO.WriteCallRead(vdeIn, "FileVDE", FileVDE_binary_dir)
	return np.moveaxis(vdeOut['weights'],-1,0),np.moveaxis(vdeOut['offsets'],[-1,-2],[0,1])



"""
a=np.array((1,2,3))
print(a)
b=V2Sym(a)
print(b)
c=Sym2V(b)
print(c)
d=np.array(((1,2),(3,4)))
print(d)
print(Sym2V(d))
"""


dim=5
bounds = (100,)
case = (1,)
A = np.random.standard_normal( (dim,dim,)+bounds )
a = A[:,:,1]

# Single decomposition
m = dotP_AA(transP(a),a)
coefs,offsets = VoronoiDecomp(m)
print(coefs)
print(offsets)
print(m) 
rec = multP(coefs,outerP(offsets,offsets)).sum(2)
print(np.max(np.abs(rec-m)))

#multiple decompositions
M = dotP_AA(transP(A),A)
Coefs,Offsets = VoronoiDecomp(M)
Rec = multP(Coefs,outerP(Offsets,Offsets)).sum(2)
print(np.max(np.abs(Rec-M)))
print(np.amin(Coefs))


# Python only alternative, works only in dimension d<=3
"""
coefs,offsets = Selling.DecompP(m)
Coefs,Offsets = Selling.DecompP(M)
"""

"""
aI = inverseP(a)
AI = inverseP(A)
print(dotP_AA(A,AI))
"""
