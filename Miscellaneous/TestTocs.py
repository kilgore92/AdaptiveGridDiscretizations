import nbformat
import os
import json
import TocTools
import sys

"""
This file, when executed as a Python script, tests that the table of contents 
of the notebooks are all correct, and reports any inconsistency.

With the optional argument --update, the table of contents are updated.
"""

def ListNotebookDirs():
	return [dirname for dirname in os.listdir() if dirname[:10]=="Notebooks_"]
def ListNotebookFiles(dirname):
	filenames_extensions = [os.path.splitext(f) for f in os.listdir(dirname)]
	return [filename for filename,extension in filenames_extensions 
	if extension==".ipynb" and filename!="Summary"]


def UpdateToc(filepath,data,cell,toc,update=False):
	if not ( ('tags' in cell['metadata'] and 'TOC' in cell['metadata']['tags'])
		or (len(cell['source'])>0 and cell['source'][0]==toc[0])): 
		return False # Not a TOC cell

	toc[-1]=toc[-1].strip()
	cell['source'][-1] = cell['source'][-1].strip()
	if toc==cell['source']:
		return True # No need to update

	print(f"TOC of file {filepath} needs updating")
	print("------- Old toc -------\n",*cell['source'])
	print("------- New toc -------\n ",*toc)
	if update:
		cell['source'] = toc
		with open(filename,'w') as f:
			json.dump(data,f,ensure_ascii=False)
	return True

def TestToc(dirname,filename,update=False):
	filepath = os.path.join(dirname,filename)+".ipynb"
	with open(filepath, encoding='utf8') as data_file:
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

	toc = TocTools.displayTOC(dirname+"/"+filename,dirname[10:]).splitlines(True)
	for c in data['cells']:
		if UpdateToc(filepath,data,c,toc,update): return
	print("directory : ",dirname," file : ",filename, " toc not found")


def TestTocs(dirname,update=False):
	filepath = os.path.join(dirname,"Summary.ipynb")
	with open(filepath, encoding='utf8') as data_file:
		data = json.load(data_file)
	toc = TocTools.displayTOCs(dirname[10:],dirname+"/").splitlines(True)
	for c in data['cells']:
		if UpdateToc(filepath,data,c,toc,update): return
	print("directory : ",dirname," Summary toc not found")

def TestTocss(update=False):
	filename = "Summary.ipynb"
	with open(filename, encoding='utf8') as data_file:
		data = json.load(data_file)
	toc = TocTools.displayTOCss().splitlines(True)
	for c in data['cells']:
		if UpdateToc(filename,data,c,toc,update): return
	print("Main Summary toc not found")


if __name__ == '__main__':
#	TestToc("Notebooks_Algo","Dense")
#	TestTocs("Notebooks_Algo")
#	TestTocss()
	update = len(sys.argv)>=2 and sys.argv[1]=='--update'

	TestTocss(update)
	for dirname in ListNotebookDirs():
		TestTocs(dirname,update)
		for filename in ListNotebookFiles(dirname):
			TestToc(dirname,filename,update)

