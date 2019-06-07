
import numpy as np
import numbers


# ----- Basic utilities functions -----

def GetGeodesics(output,suffix=''): 
	if suffix != '': suffix='_'+suffix
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

def Rect(corner0,corner1,sampleBoundary=False,gridScale=None,gridScales=None,dimx=None,dims=None):
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

# ---------------- Independent from HFM code ------------------

def FlattenSymmetricMatrix(m):
	"""
	Input : a square (symmetric) matrix.
	Output : a vector containing the lower triangular entries
	"""
	d=m.shape[0]
	assert(d==m.shape[1])
	return np.array([ m[i,j] for i in range(d) for j in range(i+1)])

# ---------------- Array ordering for import/export ----------

# Maybe this would be better located in the c++ code, IO interface. 
# Because here, we cannot deal with fields depending only on some of the coordinates.
# Left as utility, but already deprecated.
def Coordinates_Components_Transpose(hfm,move_coordinates_first):
    hfm_old = {}
    dims = tuple(hfm['dims'].astype(int))
    dim=len(dims)
    if hfm['arrayOrdering']=='YXZ_RowMajor':
        if dim==2: dims = dims[1],dims[0]
        if dim>2: dims = dims[1],dims[0],dims[2:]
        
    for key,val in hfm.items():
        if not isinstance(val,np.ndarray): continue
        vdims= val.shape
        vdim = len(vdims)
        if vdim<=dim: continue
        if move_coordinates_first:
            if not vdims[vdim-dim:vdim] == dims: continue
            hfm_old[key]=val
            hfm[key] = np.transpose(val, tuple(range(vdim-dim,vdim)) + tuple(range(vdim-dim)) )
        else: # move_coordinates_last
            if not vdims[0:dim] == dims: continue
            hfm_old[key]=val
            hfm[key] = np.transpose( val, tuple(range(dim,vdim)) + tuple(range(dim)) )
    return hfm_old
