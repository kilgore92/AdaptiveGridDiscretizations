import nbformat 
from nbconvert.preprocessors import ExecutePreprocessor,CellExecutionError

default_notebook_filenames = [
"TensorSelling","TensorVoronoi",
"MonotoneSchemes1D","LinearMonotoneSchemes2D","NonlinearMonotoneFirst2D","NonlinearMonotoneSecond2D",
"MongeAmpere","VaradhanGeodesics"
]

def TestNotebook(notebook_filename, result_path):
	print("Testing notebook " + notebook_filename)
	filename,extension = os.path.splitext(notebook_filename)
	if extension=='': extension='.ipynb'
	filename_out = filename+"_out"
	with open(filename+extension) as f:
		nb = nbformat.read(f,as_version=4) # alternatively nbformat.NO_CONVERT
	ep = ExecutePreprocessor(timeout=600,kernel_name='python3')

	try:
		out = ep.preprocess(nb,{}) #, {'metadata': {'path': run_path}}
	except CellExecutionError:
		out = None
		msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
		msg += 'See notebook "%s" for the traceback.' % filename_out+extension
		print(msg)
		raise
	finally:
		with open(result_path+filename_out+extension, mode='wt') as f:
			nbformat.write(nb, f)

if __name__ == '__main__':
	import sys
	import os
	result_path = "../test_results/"
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	notebook_filenames = sys.argv[1:] if len(sys.argv)>=2 else default_notebook_filenames
	for notebook_filename in notebook_filenames:
		TestNotebook(notebook_filename,result_path)