import xml.etree.ElementTree as ET
import xmltodict
import json
import os

tree = ET.parse(os.path.join('data', 'grade_data.xml'))
xml_data = tree.getroot()

xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')


data_dict = dict(xmltodict.parse(xmlstr))

print(data_dict)

if not os.path.exists('munge'):
    os.makedirs('munge')

with open(os.path.join('munge', 'new_data.json'), 'w+') as json_file:
    json.dump(data_dict, json_file, indent=4, sort_keys=True)