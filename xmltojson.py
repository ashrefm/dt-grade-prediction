# MIT License
#
# Copyright (c) 2019 Mohamed-Achref MAIZA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

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