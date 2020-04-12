import lxml.etree as ET
import os
dom = ET.parse("structure.xml")
xslt = ET.parse("transform.xml")
transform = ET.XSLT(xslt)
newdom = transform(dom)
os.remove("cartegrise.html")
f= open("cartegrise.html","x")
f.write(str(newdom))
f.close()
print(ET.tostring(newdom, pretty_print=True))
