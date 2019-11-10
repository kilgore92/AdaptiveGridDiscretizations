import numpy as np
import importlib

from .LibraryCall import RunDispatch,GetBinaryDir
from .run_refined import RunRefined


def Run(hfmIn):
	"""
	Raw call to the HFM library on the input parameters, returns output and prints log.
	"""
	return RunDispatch(hfmIn,GetBinaryDir("FileHFM","HFMpy"))

def VoronoiDecomposition(arr):
	"""
	Call the FileVDQ library to decompose the provided quadratic form(s).
	"""
	from ..Metrics import misc
	from . import FileIO
	bin_dir = GetBinaryDir("FileVDQ",None)
	vdqIn ={'tensors':np.moveaxis(misc.flatten_symmetric_matrix(arr),0,-1)}
	vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
	return np.moveaxis(vdqOut['weights'],-1,0),np.moveaxis(vdqOut['offsets'],[-1,-2],[0,1])


def reload_submodules():
	from importlib import reload
	import sys
	hfm = sys.modules['NumericalSchemes.HFMUtils']

	global RunDispatch
	hfm.LibraryCall = reload(hfm.LibraryCall)
	RunDispatch =  LibraryCall.RunDispatch

	global RunRefined
	hfm.run_refined = reload(hfm.run_refined)
	RunRefined =  run_refined.RunRefined


# ----- Basic utilities for HFM input and output -----

def GetGeodesics(output,suffix=''): 
	if suffix != '' and not suffix.startswith('_'): suffix='_'+suffix
	return np.vsplit(output['geodesicPoints'+suffix],
					 output['geodesicLengths'+suffix].cumsum()[:-1].astype(int))

SEModels = {'ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2',
'ReedsSheppExt2','ReedsSheppForwardExt2','ElasticaExt2','DubinsExt2',
'ReedsShepp3','ReedsSheppForward3'}

def GetCorners(params):
	dims = params['dims']
	dim = len(dims)
	h = params['gridScales'] if 'gridScales' in params.keys() else [params['gridScale']]*dim
	origin = params['origin'] if 'origin' in params.keys() else [0.]*dim
	if params['model'] in SEModels:
		origin = np.append(origin,[0]*(dim-len(origin)))		
		hTheta = 2*np.pi/dims[-1]
		h[-1]=hTheta; origin[-1]=-hTheta/2;
		if dim==5: h[-2]=hTheta; origin[-2]=-hTheta/2;
	return [origin,origin+h*dims]

def CenteredLinspace(a,b,n): 
	r,dr=np.linspace(a,b,n,endpoint=False,retstep=True)
	return r+dr/2

def GetAxes(params,dims=None):
	bottom,top = GetCorners(params)
	if dims is None: dims=params['dims']
	return [CenteredLinspace(b,t,d) for b,t,d in zip(bottom,top,dims)]

def GetGrid(params,dims=None):
	axes = GetAxes(params,dims);
	ordering = params['arrayOrdering']
	if ordering=='RowMajor':
		return np.meshgrid(*axes,indexing='ij')
	elif ordering=='YXZ_RowMajor':
		return np.meshgrid(*axes)
	else: 
		raise ValueError('Unsupported arrayOrdering : '+ordering)

def Rect(sides,sampleBoundary=False,gridScale=None,gridScales=None,dimx=None,dims=None):
	"""
	Defines a box domain, for the HFM library.
	Inputs.
	- sides, e.g. ((a,b),(c,d),(e,f)) for the domain [a,b]x[c,d]x[e,f]
	- sampleBoundary : switch between sampling at the pixel centers, and sampling including the boundary
	- gridScale, gridScales : side h>0 of each pixel (alt : axis dependent)
	- dimx, dims : number of points along the first axis (alt : along all axes)
	"""
	corner0,corner1 = np.array(sides,dtype=float).T
	dim = len(corner0)
	sb=float(sampleBoundary)
	result=dict()
	width = np.array(corner1)-np.array(corner0)
	if gridScale is not None:
		gridScales=[gridScale]*dim; result['gridScale']=gridScale
	elif gridScales is not None:
		result['gridScales']=gridScales
	elif dimx is not None:
		gridScale=width[0]/(dimx-sb); gridScales=[gridScale]*dim; result['gridScale']=gridScale
	elif dims is not None:
		gridScales=width/(np.array(dims)-sb); result['gridScales']=gridScales
	else: 
		raise ValueError('Missing argument gridScale, gridScales, dimx, or dims')

	h=gridScales
	ratios = [(M-m)/delta+sb for delta,m,M in zip(h,corner0,corner1)]
	dims = [round(r) for r in ratios]
	assert(np.min(dims)>0)
	origin = [c+(r-d-sb)*delta/2 for c,r,d,delta in zip(corner0,ratios,dims,h)]
	result.update({'dims':np.array(dims),'origin':np.array(origin)});
	return result