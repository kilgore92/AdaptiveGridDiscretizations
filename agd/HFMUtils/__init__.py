import numpy as np
import importlib


from .LibraryCall import GetBinaryDir
from .run_detail import RunRaw,RunSmart


def Run(hfmIn,smart=False,**kwargs):
	"""
	Calls to the HFM library, returns output and prints log.

	Parameters
	----------
	smart : bool  
		Choose between a smart and raw run
	**kwargs
		Passed to RunRaw or RunSmart
	"""
	return RunSmart(hfmIn,**kwargs) if smart else RunRaw(hfmIn,**kwargs)

def VoronoiDecomposition(arr):
	"""
	Calls the FileVDQ library to decompose the provided quadratic form(s),
	as based on Voronoi's first reduction of quadratic forms.
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
	hfm = sys.modules['agd.HFMUtils']

	global GetBinaryDir
	hfm.LibraryCall = reload(hfm.LibraryCall)
	GetBinaryDir =  LibraryCall.GetBinaryDir

	global RunRaw,RunSmart
	hfm.run_detail = reload(hfm.run_detail)
	RunSmart =  run_detail.RunSmart
	RunRaw = run_detail.RunRaw


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


# -------------- Point to and from index --------------

def PointFromIndex(params,index,to=False):
	"""
	Turns an index into a point.
	Optional argument to: if true, inverse transformation, turning a point into a continuous index
	"""
	bottom,top = GetCorners(params)
	dims=np.array(params['dims'])
	
	scale = (top-bottom)/dims
	start = bottom +0.5*scale
	if not to: return start+scale*index
	else: return (index-start)/scale

def IndexFromPoint(params,point):
	"""
	Returns the index that yields the position closest to a point, and the error.
	"""
	continuousIndex = PointFromIndex(params,point,to=True)
	index = np.round(continuousIndex)
	return index.astype(int),(continuousIndex-index)


# ----------- Helper class ----------

class dictIn(dict):
	"""
	A very shallow subclass of a python dictionnary, intended for storing the inputs to the HFM library.
	Usage: a number of the free functions of HFMUtils are provided as methods, for convenience.
	"""

	# Coordinates related methods
	@property
	def Corners(self):
		return GetCorners(self)
	def SetRect(self,*args,**kwargs):
		self.update(Rect(*args,**kwargs))

	Axes=GetAxes
	Grid=GetGrid
	PointFromIndex=PointFromIndex
	IndexFromPoint=IndexFromPoint

	# Running
	Run = Run
	RunRaw = RunRaw
	RunSmart = RunSmart

#	def Axes(self,*args,**kwargs):
#		return GetAxes(self,*args,**kwargs)
#	def Grid(self,*args,**kwargs):
#		return GetGrid(self,*args,**kwargs)
#	def PointFromIndex(self,*args,**kwargs):
#		return PointFromIndex(self,*args,**kwargs)
#	def IndexFromPoint(self,*args,**kwargs):
#		return IndexFromPoint(self,*args,**kwargs)

#	def Run(self,*args,**kwargs):
#		return Run(self,*args,**kwargs)
#	def RunRaw(self,*args,**kwargs):
#		return RunRaw(self,*args,**kwargs)
#	def RunSmart(self,*args,**kwargs):
#		return RunSmart(self,*args,**kwargs)






