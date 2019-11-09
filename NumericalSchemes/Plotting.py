# ------- Plotting  ------
from os import path

def SetTitle3D(ax,title):
	ax.text2D(0.5,0.95,title,transform=ax.transAxes,horizontalalignment='center')

def savefig(fig,fileName,dirName=None,**kwargs):
	"""Save a figure with:
- in a given directory, possibly set in the properties of the function. 
 Silently fails if dirName is None
- defaulted arguments, possibly set in the properties of the function
"""
	# Set arguments to be passed
	for key,value in vars(savefig).items():
		if key not in kwargs and key!='dirName':
			kwargs[key]=value

	# Set directory
	if dirName is None: 
		if savefig.dirName is None: return 
		else: dirName=savefig.dirName
	
	# Save figure
	if path.isdir(dirName):
		fig.savefig(path.join(dirName,fileName),**kwargs) 
	else:
		print("savefig error: No such directory", dirName)
#		raise OSError(2, 'No such directory', dirName)

savefig.dirName = None
savefig.bbox_inches = 'tight'
savefig.pad_inches = 0
savefig.dpi = 300