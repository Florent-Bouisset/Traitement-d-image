from io import StringIO
from lxml import etree

f = open("structure.xml", "rb")
xml = f.read()
dtd = etree.DTD("structure.dtd")
root = etree.XML(xml)
print(dtd.validate(root))

print(dtd.error_log.filter_from_errors())
