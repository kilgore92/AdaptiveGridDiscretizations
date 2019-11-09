import numpy as np
import numbers
import importlib
import ast

def SetInput(hfm,params):
	for key,val in params.items():
		if isinstance(val,numbers.Number):
			hfm.set_scalar(key,val)
		elif isinstance(val,str):
			hfm.set_string(key,val)
		elif isinstance(val,np.ndarray):
			hfm.set_array(key,val)
		else:
			raise ValueError('Invalid type for key ' + key);

def GetOutput(hfm):
	comp=hfm.computed_keys()
	if comp[0]=='(': # Should be patched now
		comp=comp.replace('),)',"']]")
		comp=comp.replace('),(',')(')
		comp=comp.replace(',',"','")
		comp=comp.replace(')(',"'],['")
		comp=comp.replace('((',"[['")

	result = {}
	for key,t in ast.literal_eval(comp):
		if t=='float':
			result[key] = hfm.get_scalar(key)
		elif t=='string':
			result[key] = hfm.get_string(key)
		elif t=='array':
			result[key] = hfm.get_array(key)
		else:
			raise ValueError('Unrecognized type '+ t + ' for key '+ key)
	return result

def ListToNDArray(params):
	for (key,val) in params.items():
		if isinstance(val,list):
			params[key]=np.array(val)

def RunDispatch(params,bin_dir):
	modelName = params['model']
	ListToNDArray(params)
	if bin_dir is None:
		moduleName = 'HFMpy.HFM_'+modelName
		HFM = importlib.import_module(moduleName)
		hfm = HFM.HFMIO()
		SetInput(hfm,params)
		hfm.run()
		return GetOutput(hfm)
	else:
		from . import FileIO
		execName = 'FileHFM_'+modelName
		return FileIO.WriteCallRead(params, execName, bin_dir)


