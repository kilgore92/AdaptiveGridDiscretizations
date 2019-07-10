# Inspired by https://fr.slideshare.net/jimarlow
import json
from IPython.display import display, Markdown, HTML

def ViewOnline(inFName,volume):
	subdir = "" if volume is None else "Notebooks_"+volume+"/"
	return ("[(view online)]("+
		"http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/"+
		subdir+inFName+".ipynb"+")")

def MakeLink(inFName,volume):
	dirName = "./Notebooks_"+volume+"/"; extension = ".ipynb"
	display(Markdown("Notebook ["+inFName+"]("+dirName+inFName+extension+") "+ViewOnline(inFName,volume)
		+ ", from volume "+ volume + " [Summary]("+dirName+"Summary"+extension+") "+ViewOnline("Summary",volume) ))

def Info(volume):
	if volume in ['NonDiv','Div','Algo']:
		return """
**Acknowledgement.** The experiments presented in these notebooks are part of ongoing research, 
some of it with PhD student Guillaume Bonnet, in co-direction with Frederic Bonnans.

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""
	elif volume=='FMM':
		return """
This Python&reg; notebook is intended as documentation and testing for the [HamiltonFastMarching (HFM) library](https://github.com/mirebeau/HamiltonFastMarching), which also has interfaces to the Matlab&reg; and Mathematica&reg; languages. 
More information on the HFM library in the manuscript:
* Jean-Marie Mirebeau, Jorg Portegies, "Hamiltonian Fast Marching: A numerical solver for anisotropic and non-holonomic eikonal PDEs", 2019 [(link)](https://hal.archives-ouvertes.fr/hal-01778322)

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""

VolumeFilenames = {
'FMM':[
    "Isotropic","Riemannian","Rander","AsymmetricQuadratic",
    "Curvature","Curvature3","DeviationHorizontality",
    "HighAccuracy","Sensitivity",
    "Illusion","Tubular","FisherRao","DubinsZermelo"
],
'NonDiv':[
	"MonotoneSchemes1D","LinearMonotoneSchemes2D","NonlinearMonotoneFirst2D","NonlinearMonotoneSecond2D",
	"MongeAmpere","OTBoundary1D","EikonalEulerian"
],
'Div':["Elliptic","EllipticAsymmetric","VaradhanGeodesics"],
'Algo':["TensorSelling","TensorVoronoi","Sparse"],
}

RepositoryDescription = """
**Github repository** to run and modify the examples on your computer.
[AdaptiveGridDiscretizations](https://github.com/Mirebeau/AdaptiveGridDiscretizations)\n
"""

def displayTOC(inFName,volume):
	with open(inFName+".ipynb") as data_file:
		data = json.load(data_file)
	contents = []
	for c in data['cells']:
		s=c['source']
		if len(s)==0:
			continue
		line1 = s[0].strip()
		if line1.startswith('#'):
			count = line1.count('#')-1
			plainText = line1[count+1:].strip()
			if plainText[0].isdigit() and int(plainText[0])!=0:
				link = plainText.replace(' ','-')
				listItem = "  "*count + "* [" + plainText + "](#" + link + ")"
				contents.append(listItem)

	display(Markdown("[**Summary**](Summary.ipynb) of this series of notebooks. "+ViewOnline(inFName,volume)))
	display(Markdown("""[**Main summary**](../Summary.ipynb), including the other volumes of this work. """ +ViewOnline("Summary",None) ))
#	display(HTML("<a id = 'table_of_contents'></a>"))
	display(Markdown("\n# Table of contents"))
	display(Markdown("\n".join(contents)))
	display(Markdown("\n\n"+Info(volume)))

def displayTOCs(volume):
	inFNames = VolumeFilenames[volume]
	contents = []
	part = ""
	part_counter = 0
	part_numerals = "ABCDEFGHIJK"
	chapter_counter = 0
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	for _inFName in inFNames:
		inFName = _inFName+".ipynb"
		with open(inFName) as data_file:
			data = json.load(data_file)
			# Display the chapter
			s=data['cells'][0]['source']
			sec = s[2][len("## Part : "):]
			if sec!=part:
				part=sec
				contents.append("### "+part_numerals[part_counter]+". "+part)
				part_counter+=1
				chapter_counter=0
			else:
				chapter_counter+=1
			chapter = s[3][len("## Chapter : "):].strip()
			contents.append(" " + "* "+chapter_numerals[chapter_counter] +
				". [" + chapter + "](" + inFName + ")" +
				" " + ViewOnline(_inFName,volume) + "\n")
			# Display the sub chapters
			for c in data['cells']:
				s = c['source']
				if len(s)==0: 
					continue
				line1 = s[0].strip()
				if line1.startswith('##') and line1[3].isdigit() and int(line1[3])!=0:
					contents.append(" "*2 + line1[len("## "):]+"\n")
			contents.append("\n")

	display(Markdown(RepositoryDescription))
	display(Markdown("# Table of contents"))
	display(Markdown("""[**Main summary**](../Summary.ipynb), including the other volumes of this work. """ +ViewOnline("Summary",None) ))
	display(Markdown("\n".join(contents)))

def displayTOCss():
	extension = '.ipynb'
	contents = []
	volume_numerals = "1234"
	part_numerals = "ABCDEFGHIJK"
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	for volume_counter,volume in enumerate(['FMM','NonDiv','Div','Algo']):
		dirName = 'Notebooks_'+volume+'/'
		part = ""
		part_counter = 0

		inFName = dirName+'Summary'+extension
		with open(inFName) as data_file:
			data = json.load(data_file)
			s = data['cells'][0]['source']
			volumeTitle = s[2][len("# Volume : "):]
			contents.append("### " + volume_numerals[volume_counter]+". "+
				"["+volumeTitle+"]("+inFName+")"+
				" "+ViewOnline('Summary',volume))

		# Display parts and chapters
		for _inFName in VolumeFilenames[volume]:
			inFName = dirName+_inFName+extension
			with open(inFName) as data_file:
				data = json.load(data_file)
				# Display the chapter
				s=data['cells'][0]['source']
				sec = s[2][len("## Part : "):]
				if sec!=part:
					part=sec
					contents.append(" * "+part_numerals[part_counter]+". "+part)
					part_counter+=1
					chapter_counter=0
				else:
					chapter_counter+=1
				chapter = s[3][len("## Chapter : "):].strip()
				contents.append("  " + "* "+chapter_numerals[chapter_counter] +
					". [" + chapter + "](" + inFName + ")" +
					" " + ViewOnline(_inFName,volume) + "\n")
	display(Markdown("# Table of contents"))
	display(Markdown("\n".join(contents)))







