import numpy as np
from NumericalSchemes import FileIO
import os

def FlattenSymmetricMatrix(a):
	"""
		Extract the lower triangular part of a (symmetric) matrix, put in a vector
	"""
	m,n = a.shape[:2]
	if m!=n:
		raise ValueError("VDEUtils.FlattenSymmetricMatrix error: matrix is not square")
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
		Call the FileVDZ library to decompose the provided tensor field.
	"""
	bin_dir = FileVDE_binary_dir if 'FileVDE_binary_dir' in globals() else GetFileVDE_binary_dir()
	vdeIn ={'tensors':np.moveaxis(FlattenSymmetricMatrix(a),0,-1)}
	vdeOut = FileIO.WriteCallRead(vdeIn, "FileVDE", bin_dir)
	return np.moveaxis(vdeOut['weights'],-1,0),np.moveaxis(vdeOut['offsets'],[-1,-2],[0,1])

def GetFileVDE_binary_dir():
	set_directory_msg = """
IMPORTANT : Please set the path to the FileVDE compiled binaries, as follows : \n
>>> VDEUtils.FileVDE_binary_dir = "path/to/FileVDE/bin"\n
\n
In order to do this automatically in the future, please set this path 
in the first line of a file named 'FileVDE_binary_dir.txt' in the current directory\n
>>> with open('FileVDE_binary_dir.txt','w+') as file: file.write("path/to/FileVDE/bin")
"""
	try:
		with open('FileVDE_binary_dir.txt','r') as f:
			FileVDE_binary_dir = f.readline().replace('\n','')
			if not os.path.isdir(FileVDE_binary_dir):
				print("ERROR : the path to the FileVDE binaries appears to be incorrect.\n")
				print("Current path : ", FileVDE_binary_dir, "\n")
				print(set_directory_msg)
			return FileVDE_binary_dir
	except OSError as e:
		print("ERROR : the path to the FileVDE binaries is not set\n")
		print(set_directory_msg)