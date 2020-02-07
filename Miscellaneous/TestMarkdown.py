import os
import sys
import json
"""
This file tests for an annoying Markdown typesetting issue: 
a mathematical expression of the form
$$
	a
	+ b
$$
will render correctly on some markdown viewers (including the jupyter notebook editor), 
and not on others (including nbviewer).

The solution is to replace + with {+} in the start of a line, 
if the next character is a blank space. Likewise for - and *.

---
Another issue was encountered in an equation of the form
$$
<n
$$

--- 
Another issue occurs when a line contains a single starting comment tag, 
and the previous line is not blank.

blabla.
<!---
my comment
--->

"""

def TestMath(filepath,update=False,show=False):
	with open(filepath, encoding='utf8') as data_file:
		data = json.load(data_file)
	for cell in data["cells"]:
		if cell['cell_type']!='markdown': continue
		eqn = None
		prevLine="\n"
		for line in cell['source']:
			if line=="$$" or line=="$$\n":
				eqn = "" if eqn is None else None
				continue
			if eqn is not None:
				eqn = eqn+line
				l = line.lstrip()
				if line[0]=='<' or (l[0] in ['+','-','*'] and l[1]==' '):
					print(f"--- Markdown displaymath issue in file {filepath} : ---")
					print(eqn)
					if show: print("(Cell contents) : \n", *cell["source"])
			if line==("<!---\n") and prevLine!="\n":
				print(f"--- Markdown comment issue in file {filepath} : ---")
				print([prevLine],line)
				if show: print("(Cell contents) : \n", *cell["source"])
			prevLine = line


if __name__ == '__main__':
#	TestToc("Notebooks_Algo","Dense")
#	TestTocs("Notebooks_Algo")
#	TestTocss()
	kwargs = {"update":False,"show":False}
	for key in sys.argv[1:]:
		assert key[:2]=="--" and key[2:] in kwargs
		kwargs[key[2:]]=True

	for dirname in os.listdir():
		if not dirname.startswith("Notebooks_"): continue
		for filename in os.listdir(dirname):
			if not filename.endswith(".ipynb"): continue
			TestMath(os.path.join(dirname,filename),**kwargs)
