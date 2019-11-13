import nbformat
import os
import json
import TocTools

def ListNotebookDirs():
	return [dirname for dirname in os.listdir() if dirname[:10]=="Notebooks_"]
def ListNotebookFiles(dirname):
	filenames_extensions = [os.path.splitext(f) for f in os.listdir(dirname)]
	return [filename for filename,extension in filenames_extensions if extension==".ipynb" and filename!="Summary"]

def TestToc(dirname,filename):
	with open(dirname+"/"+filename+".ipynb", encoding='utf8') as data_file:
		data = json.load(data_file)

	# Test Header
	s = data['cells'][0]['source']
	line0 = s[0].strip()
	line0_ref = (
		"# The HFM library - A fast marching solver with adaptive stencils"
		if dirname=="Notebooks_FMM" else
		"# Adaptive PDE discretizations on cartesian grids")

	if line0!=line0_ref:
		print("directory : ",dirname," file : ",filename,
			" line0 : ",line0," differs from expexted ",line0_ref)

	line1 = s[1].strip()
	line1_ref = {
	'Notebooks_Algo':	"## Volume: Algorithmic tools",
	'Notebooks_Div':	"## Volume: Divergence form PDEs",
	'Notebooks_NonDiv':	"## Volume: Non-divergence form PDEs",
	'Notebooks_FMM':	"",
	'Notebooks_Repro':	"## Volume: Reproducible research",
	}[dirname]

	if line0!=line0_ref:
		print("directory : ",dirname," file : ",filename,
			" line1 : ",line1," differs from expexted ",line1_ref)

	toc = TocTools.displayTOC(dirname+"/"+filename,dirname[10:]).strip()
	for c in data['cells']:
		s = "".join(c['source']).strip()
		if s[:20]==toc[:20]:
			if s!=toc:
				print("directory : ",dirname," file : ",filename,
				" toc needs updating")
#				print(s)
#				print(toc)
			return
	print("directory : ",dirname," file : ",filename, " toc not found")




def TestTocs(dirname):
	with open(dirname+"/Summary.ipynb", encoding='utf8') as data_file:
		data = json.load(data_file)
	toc = TocTools.displayTOCs(dirname[10:],dirname+"/").strip()
	for c in data['cells']:
		s = "".join(c['source']).strip()
		if s[:20]==toc[:20]:
			if s!=toc:
				print("directory : ",dirname," Summary toc needs updating")
#				print(s)
#				print(toc)
			return
	print("directory : ",dirname," Summary toc not found")

def TestTocss():
	with open("Summary.ipynb", encoding='utf8') as data_file:
		data = json.load(data_file)
	toc = TocTools.displayTOCss().strip()
	for c in data['cells']:
		s = "".join(c['source']).strip()
		if s[:20]==toc[:20]:
			if s!=toc:
				print("Main Summary toc needs updating")
#				print(s)
#				print(toc)
			return
	print("Main Summary toc not found")



if __name__ == '__main__':
#	TestToc("Notebooks_Algo","Dense")
#	TestTocs("Notebooks_Algo")
#	TestTocss()


	TestTocss()
	for dirname in ListNotebookDirs():
		TestTocs(dirname)
		for filename in ListNotebookFiles(dirname):
			TestToc(dirname,filename)

