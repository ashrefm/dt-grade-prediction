"""Converts to Dt-Grade XML data to json format."""

import xml.etree.ElementTree as ET
import xmltodict
import json
import os


def main():

    file = os.path.join('data', 'grade_data.xml')
    if not os.path.exists(file):
        raise IOError("File does not exist: %s" % file)
    
    tree = ET.parse(file)
    xml_data = tree.getroot()
    xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')
    data_dict = dict(xmltodict.parse(xmlstr))

    if not os.path.exists('munge'):
        os.makedirs('munge')

    with open(os.path.join('munge', 'grade_data.json'), 'w+') as json_file:
        json.dump(data_dict, json_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()