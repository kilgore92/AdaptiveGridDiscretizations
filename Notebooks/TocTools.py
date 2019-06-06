# Adapted from https://fr.slideshare.net/jimarlow
import json
from IPython.display import display, Markdown, HTML


def displayTOC(inFName):
	text_begin = """
[**Main summary**](Summary.ipynb)
of this series of notebooks. 
[(view online)](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks/Summary.ipynb)
"""

	text_end = """
**Acknowledgement.** The experiments presented in this notebook are part of ongoing research, 
some of it with PhD student Guillaume Bonnet, in co-direction with Frederic Bonnans.

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""

	with open(inFName) as data_file:
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
				listItem = " "*count + "* [" + plainText + "](#" + link + ")"
				contents.append(listItem)
	display(Markdown(text_begin))
#	display(HTML("<a id = 'table_of_contents'></a>"))
	display(Markdown("\n# Table of contents"))
	display(Markdown("\n".join(contents)))
	display(Markdown("\n\n"+text_end))

def displayTOCs(inFNames):
	contents = []
	section = ""
	section_counter = 0
	section_numerals = "ABCDEFGHIJK"
	chapter_counter = 0
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	nbviewer = "http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/HFM_Python_Notebooks/master/"
	for _inFName in inFNames:
		inFName = _inFName+".ipynb"
		with open(inFName) as data_file:
			data = json.load(data_file)
			# Display the chapter
			s=data['cells'][0]['source']
			sec = s[2][len("## Part : "):]
			if sec!=section:
				section=sec
				contents.append("### "+section_numerals[section_counter]+". "+section)
				section_counter+=1
				chapter_counter=0
			else:
				chapter_counter+=1
			chapter = s[3][len("## Chapter : "):].strip()
			contents.append(" " + "* "+chapter_numerals[chapter_counter] +
				". ["+ chapter +"](" +inFName + ")"
				" [(view online)](" + nbviewer + inFName +")\n")
			# Display the sub chapters
			for c in data['cells']:
				s = c['source']
				if len(s)==0: 
					continue
				line1 = s[0].strip()
				if line1.startswith('##') and line1[3].isdigit() and int(line1[3])!=0:
					contents.append(" "*2 + line1[len("## "):]+"\n")
			contents.append("\n")
	display(Markdown("# Table of contents"))
	display(Markdown("\n".join(contents)))

