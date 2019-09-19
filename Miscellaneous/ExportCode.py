import nbformat 
import json
import sys
import os


result_path = "ExportedCode"

def ListNotebooks():
	filenames_extensions = [os.path.splitext(f) for f in os.listdir()]
	return [filename for filename,extension in filenames_extensions if extension==".ipynb"]

def ExportCode(inFName,outFName):
	with open(inFName) as data_file:
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
		print("Exporting ", nAlgo, " code from notebook ", inFName, " in file ", outFName)
		with open(outFName,'w+') as output_file:			
			for c in output:
				output_file.write(c)

if __name__ == '__main__':
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	notebook_filenames = sys.argv[1:] if len(sys.argv)>=2 else ListNotebooks()

	for name in notebook_filenames:
		ExportCode(name+'.ipynb',result_path+"/"+name+'.py')


