import numpy as np
import os

from . import FileIO
from . import GetBinaryDir

def FlattenSymmetricMatrix(a):
	"""
		Extract the lower triangular part of a (symmetric) matrix, put in a vector
	"""
	m,n = a.shape[:2]
	if m!=n:
		raise ValueError("VDQUtils.FlattenSymmetricMatrix error: matrix is not square")
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

def Decomposition(a):
	"""
		Call the FileVDQ library to decompose the provided quadratic form(s).
	"""
	bin_dir = GetBinaryDir("FileVDQ",None)
	vdqIn ={'tensors':np.moveaxis(FlattenSymmetricMatrix(a),0,-1)}
	vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
	return np.moveaxis(vdqOut['weights'],-1,0),np.moveaxis(vdqOut['offsets'],[-1,-2],[0,1])
