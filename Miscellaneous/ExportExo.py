import nbformat 
import json
import sys
import os
import shutil

def MakeExo(FileName,ExoName):
	with open(FileName, encoding='utf8') as data_file:
		data=json.load(data_file)
	newcells = []
	for c in data['cells']:
		if 'tags' in c['metadata']:
			tags = c['metadata']['tags']
			if 'ExoRemove' in c['metadata']['tags']:
				continue
			elif 'ExoMarkdown' in tags:
				c['source']=c['source'][1:-1]
			elif 'ExoCode' in tags:
				c['cell_type']='code'
				c['source']=c['source'][1:-1]
				c['outputs'] = []
				c['execution_count'] = None
		newcells.append(c)
	data['cells']=newcells

	with open(ExoName,'w') as f:
		json.dump(data,f,ensure_ascii=False)

if __name__ == '__main__':
	for name in sys.argv[1:]:
		dir,FileName = os.path.split(name)
		prefix,ext = os.path.splitext(FileName)
		ExoName = os.path.join(dir,"Exo",prefix+'_Exo.ipynb')
#		shutil.copy(file,ExoName)
		MakeExo(FileName,ExoName)