#!/usr/bin/python
#
# Classes to store, read, and write annotations
#

import os
import json
from collections import namedtuple

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])

# Class that contains the information of a single annotated object
class CsObject:
    # Constructor
    def __init__(self):
        # the label
        self.label    = ""
        # the polygon as list of points
        self.polygon  = []

    def __str__(self):
        polyText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    polyText += '({},{}) '.format( p.x , p.y )
            else:
                polyText += '({},{}) ({},{}) ... ({},{}) ({},{})'.format(
                    self.polygon[ 0].x , self.polygon[ 0].y ,
                    self.polygon[ 1].x , self.polygon[ 1].y ,
                    self.polygon[-2].x , self.polygon[-2].y ,
                    self.polygon[-1].x , self.polygon[-1].y )
        else:
            polyText = "none"
        text = "Object: {} - {}".format( self.label , polyText )
        return text

    def fromJsonText(self, jsonText):
        self.label = str(jsonText['label'])
        self.polygon = [ Point(p[0],p[1]) for p in jsonText['polygon'] ]

# The annotation of a whole image
class Annotation:
    # Constructor
    def __init__(self):
        # the width of that image and thus of the label image
        self.imgWidth  = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects
        self.objects = []

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth  = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects   = []
        for objIn in jsonDict[ 'objects' ]:
            obj = CsObject()
            obj.fromJsonText(objIn)
            self.objects.append(obj)

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print 'Given json file not found: {}'.format(jsonFile)
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

# a dummy example
if __name__ == "__main__":
    obj = CsObject()
    obj.label = 'car'
    obj.polygon.append( Point( 0 , 0 ) )
    obj.polygon.append( Point( 1 , 0 ) )
    obj.polygon.append( Point( 1 , 1 ) )
    obj.polygon.append( Point( 0 , 1 ) )

    print obj