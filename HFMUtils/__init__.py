from HFMUtils.Geometry import *
from HFMUtils.Plotting import *
import os

from HFMUtils.LibraryCall import RunDispatch
def Run(params):
	out = RunDispatch(params,GetBinaryDir("FileHFM"))
	if 'log' in out: 
		print(out['log'])
	return out

def GetBinaryDir(libName):
	dirName = libName + "_binary_dir"
	if dirName in globals(): return globals()[dirName]
	fileName = dirName + ".txt"
	pathExample = "path/to/"+libName+"/bin"
	set_directory_msg = """
IMPORTANT : Please set the path to the """ + libName + """ compiled binaries, as follows : \n
>>> """+__name__+"."+ dirName+ """ = '""" +pathExample+ """'\n
\n
In order to do this automatically in the future, please set this path 
in the first line of a file named '"""+fileName+"""' in the current directory\n
>>> with open('"""+fileName+"""','w+') as file: file.write('"""+pathExample+"""')
"""
	try:
		with open(fileName,'r') as f:
			binary_dir = f.readline().replace('\n','')
			if binary_dir=="None":
				return None
			if not os.path.isdir(binary_dir):
				print("ERROR : the path to the "+libName+" binaries appears to be incorrect.\n")
				print("Current path : ", binary_dir, "\n")
				print(set_directory_msg)
			return binary_dir
	except OSError as e:
		print("ERROR : the path to the "+libName+" binaries is not set\n")
		print(set_directory_msg)
		raise