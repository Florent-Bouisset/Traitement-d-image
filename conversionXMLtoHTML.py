import lxml.etree as ET
import os

pathInputXML = "generatedXML.xml"
pathInputXSLT = "transform.xml"
pathOutputHTML = "cartegrise.html"

dom = ET.parse(pathInputXML)
xslt = ET.parse("transform.xml")
transform = ET.XSLT(xslt)
newdom = transform(dom)
if (os.path.exists(pathOutputHTML)):
    os.remove(pathOutputHTML)
f= open(pathOutputHTML,encoding = 'utf-8',mode = 'x')
f.write(str(newdom))
f.close()
print(ET.tostring(newdom, pretty_print=True))
os.system("start " + pathOutputHTML)
