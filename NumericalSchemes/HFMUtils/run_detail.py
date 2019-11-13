import numpy as np
from .LibraryCall import RunDispatch,GetBinaryDir
from ..Metrics.base import Base as MetricsBase


def RunRaw(hfmIn):
	"""Raw call to the HFM library"""
	return RunDispatch(hfmIn,GetBinaryDir("FileHFM","HFMpy"))

def PreProcess(key,value,raw_out):
	"""
	copies key,val from refined to raw, with adequate treatment
	"""
	if isinstance(value,MetricsBase):
		assert(key in ['cost','speed','metric','dualMetric'])
		raw_out[key] = value.to_HFM()
	else:
		raw_out[key] = value

def PostProcess(key,value,refined_out,raw_in):
	"""
	copies key,val from raw to refined, with adequate treatment
	"""
	if key.startswith('geodesicPoints'):
		from ..HFMUtils import GetGeodesics
		suffix = key[len('geodesicPoints'):]
		geodesics = GetGeodesics(raw_in,suffix=suffix)
		refined_out["geodesics"+suffix]=[np.moveaxis(geo,-1,0) for geo in geodesics]
	else:
		refined_out[key]=value

def RunSmart(hfmIn,tupleIn=tuple(),tupleOut=None):
	"""
	Calls the HFM library, with pre-processing and post-processing of data.

	tupleIn and tupleOut are intended to make the inputs and outputs 
	visible to reverse automatic differentiation
	- tupleIn : arguments specified as ((key_1,value_1), ..., (key_n,value_m))
		take precedence over similar keys hfmIn
	- tupleOut : (key_1, ..., key_n) corresponding to results 
		(value_1,...,value_n) to be return in a tuple
	"""
	
	#TODO : 
	#	- geometryFirst (default : all but seeds)
	#	- Handling of AD information, forward and reverse
	hfmIn_raw = {}

	# Pre-process tuple arguments
	tupleInKeys = {key for key,_ in tupleIn}
	if len(tupleInKeys)!=len(tupleIn):
		raise ValueError("RunProcessed error : duplicate keys in tupleIn")
	
	for key,value in tupleIn:
		PreProcess(key,value,hfmIn_raw)

	# Pre-process usual arguments
	for key,value in hfmIn.items():
		if key not in tupleInKeys:
			PreProcess(key,value,hfmIn_raw)

	hfmOut_raw = RunDispatch(hfmIn_raw,GetBinaryDir("FileHFM","HFMpy"))

	hfmOut = {'raw':hfmOut_raw}

	# Post process
	for key,val in hfmOut_raw.items():
		PostProcess(key,val,hfmOut,hfmOut_raw)

	# Extract tuple arguments
	if tupleOut is None:
		return hfmOut
	else:
		tupleResult = tuple(hfmOut.pop(key) for key in tupleOut)
		return hfmOut,tupleResult
	



