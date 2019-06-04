# Adapted from https://fr.slideshare.net/jimarlow
import json
from IPython.display import display, Markdown, HTML

text_reference = """
*Summary of this series of notebooks:*
[Adaptive grid discretizations, summary](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks/Summary.ipynb)

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""

def displayTOC(inFName):
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
	display(HTML("<a id = 'table_of_contents'></a>"))
	display(Markdown("# Table of contents"))
	display(Markdown("\n".join(contents)))
	display(Markdown("\n\n"+text_reference))