import nbformat 
import json
import sys
import os

from TestNotebooks import ListNotebooks

result_path = "ExportedCode"

def ExportCode(inFName,outFName):
	with open(inFName, encoding='utf8') as data_file:
		data = json.load(data_file)
	output = [
		'# Code automatically exported from notebook '+inFName,
		'# Do not modify',
		'import sys; sys.path.append("../..") # Allow imports from parent directory\n\n'
	]
	nAlgo = 0
	for c in data['cells']:
		if 'tags' in c['metadata'] and 'ExportCode' in c['metadata']['tags']:
			output.extend(c['source'])
			output.append('\n\n')
			nAlgo+=1
	if nAlgo>0:
		print("Exporting ", nAlgo, " code cells from notebook ", inFName, " in file ", outFName)
		with open(outFName,'w+', encoding='utf8') as output_file:			
			for c in output:
				output_file.write(c)

if __name__ == '__main__':
	notebook_filenames = sys.argv[1:] if len(sys.argv)>=2 else ListNotebooks()

	for name in notebook_filenames:
		subdir,fname = os.path.split(name)
		ExportCode(name+'.ipynb',os.path.join(subdir,result_path,fname)+'.py')


