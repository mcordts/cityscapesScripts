#!/usr/bin/python
#
# Classes to store, read, and write annotations
#

from __future__ import print_function, absolute_import, division
import os
import json
import numpy as np
from collections import namedtuple

# get current date and time
import datetime
import locale

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])

from abc import ABCMeta, abstractmethod

# Type of an object
class CsObjectType():
    POLY = 1 # polygon
    BBOX = 2 # bounding box
    BBOX3D = 3 # 3d bounding box
    IGNORE2D = 4 # 2d ignore region

# Abstract base class for annotation objects
class CsObject:
    __metaclass__ = ABCMeta

    def __init__(self, objType):
        self.objectType = objType
        # the label
        self.label    = ""

        # If deleted or not
        self.deleted  = 0
        # If verified or not
        self.verified = 0
        # The date string
        self.date     = ""
        # The username
        self.user     = ""
        # Draw the object
        # Not read from or written to JSON
        # Set to False if deleted object
        # Might be set to False by the application for other reasons
        self.draw     = True

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def fromJsonText(self, jsonText, objId=-1): pass

    @abstractmethod
    def toJsonText(self): pass

    def updateDate( self ):
        try:
            locale.setlocale( locale.LC_ALL , 'en_US.utf8' )
        except locale.Error:
            locale.setlocale( locale.LC_ALL , 'en_US' )
        except locale.Error:
            locale.setlocale( locale.LC_ALL , 'us_us.utf8' )
        except locale.Error:
            locale.setlocale( locale.LC_ALL , 'us_us' )
        except:
            pass
        self.date = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # Mark the object as deleted
    def delete(self):
        self.deleted = 1
        self.draw    = False

# Class that contains the information of a single annotated object as polygon
class CsPoly(CsObject):
    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.POLY)
        # the polygon as list of points
        self.polygon    = []
        # the object ID
        self.id         = -1

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

    def fromJsonText(self, jsonText, objId):
        self.id = objId
        self.label = str(jsonText['label'])
        self.polygon = [ Point(p[0],p[1]) for p in jsonText['polygon'] ]
        if 'deleted' in jsonText.keys():
            self.deleted = jsonText['deleted']
        else:
            self.deleted = 0
        if 'verified' in jsonText.keys():
            self.verified = jsonText['verified']
        else:
            self.verified = 1
        if 'user' in jsonText.keys():
            self.user = jsonText['user']
        else:
            self.user = ''
        if 'date' in jsonText.keys():
            self.date = jsonText['date']
        else:
            self.date = ''
        if self.deleted == 1:
            self.draw = False
        else:
            self.draw = True

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['id'] = self.id
        objDict['deleted'] = self.deleted
        objDict['verified'] = self.verified
        objDict['user'] = self.user
        objDict['date'] = self.date
        objDict['polygon'] = []
        for pt in self.polygon:
            objDict['polygon'].append([pt.x, pt.y])

        return objDict

# Class that contains the information of a single annotated object as bounding box
class CsBbox(CsObject):
    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX)
        # the polygon as list of points
        self.bbox  = []
        self.bboxVis  = []

        # the ID of the corresponding object
        self.instanceId = -1

    def __str__(self):
        bboxText = ""
        bboxText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox[0] , self.bbox[1] ,  self.bbox[2] ,  self.bbox[3] )

        bboxVisText = ""
        bboxVisText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bboxVis[0] , self.bboxVis[1] , self.bboxVis[2], self.bboxVis[3] )

        text = "Object: {} - bbox {} - visible {}".format( self.label , bboxText, bboxVisText )
        return text

    def fromJsonText(self, jsonText, objId=-1):
        self.bbox = jsonText['bbox']
        self.bboxVis = jsonText['bboxVis']
        self.label = str(jsonText['label'])
        self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['bbox'] = self.bbox
        objDict['bboxVis'] = self.bboxVis

        return objDict

# Class that contains the information of a single annotated object as 3D bounding box
class CsBbox3d(CsObject):
    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX3D)

        # modal and amodal refer to bbox and bboxVis
        self.box_2d_amodal = []
        self.box_2d_modal = []

        self.center = []
        self.dims = []
        self.rotation = []
        self.label = []
        self.score = []

    def __str__(self):
        bbox2dText = ""
        bbox2dText += 'Modal 2D:  xmin: {}, ymin: {}, xmax: {}, ymax: {}'.format(
            self.box_2d_modal[0], self.box_2d_modal[1], self.box_2d_modal[2], self.box_2d_modal[3])
        bbox2dText += 'Amodal 2D: xmin: {}, ymin: {}, xmax: {}, ymax: {}'.format(
            self.box_2d_amodal[0], self.box_2d_amodal[1], self.box_2d_amodal[2], self.box_2d_amodal[3])

        bbox3dText = ""
        bbox3dText += 'Center (x/y/z) [m]: {}/{}/{}'.format(
            self.center[0], self.center[1],  self.center[2])
        bbox3dText += 'Dimensions (l/w/h) [m]: {}/{}/{}'.format(
            self.dims[0], self.dims[1],  self.dims[2])
        bbox3dText += 'Rotation: {}/{}/{}/{}'.format(
            self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])


        text = "Object: {} - 2D {} - 3D {}".format(self.label, bbox2dText, bbox3dText)
        return text

    def fromJsonText(self, jsonText):
        self.box_2d_amodal = jsonText["2d"]["amodal"]

        # if no modal 2d box is provided, use amodal one instead
        if "modal" in jsonText["2d"].keys():
            self.box_2d_modal = jsonText["2d"]["modal"]
        else:
            self.box_2d_modal = jsonText["2d"]["amodal"]

        self.center = jsonText["3d"]["center"]
        self.dims = jsonText["3d"]["dimensions"]
        self.rotation = jsonText["3d"]["rotation"]
        self.class_name = jsonText["class_name"]
        self.score = jsonText["score"]

    def toJsonText(self):
        objDict = {}
        objDict["class_name"] = self.label
        objDict["2d"]["amodal"] = self.box_2d_amodal
        objDict["2d"]["modal"] = self.box_2d_modal
        objDict["3d"]["center"] = self.center
        objDict["3d"]["dimensions"] = self.dims
        objDict["3d"]["rotation"] = self.rotation

        return objDict

    @property
    def depth(self):
        return np.sqrt(self.center[0]**2 + self.center[2]**2).astype(int)

# Class that contains the information of a single annotated 2d ignore region
class CsIgnore2d(CsObject):
    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.IGNORE2D)

        # modal and amodal refer to bbox and bboxVis
        self.box_2d = []

    def __str__(self):
        bbox2dText = ""
        bbox2dText += 'Ignore Region:  xmin: {}, ymin: {}, xmax: {}, ymax: {}'.format(
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])

        return text

    def fromJsonText(self, jsonText):
        self.box_2d = jsonText["2d"]

# The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox)
class Annotation:
    # Constructor
    def __init__(self, objType=CsObjectType.POLY):
        # the width of that image and thus of the label image
        self.imgWidth  = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects
        self.objects = []
        assert objType in CsObjectType.__dict__.values()
        self.objectType = objType

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth  = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects   = []
        for objId, objIn in enumerate(jsonDict[ 'objects' ]):
            if self.objectType == CsObjectType.POLY:
                obj = CsPoly()
            elif self.objectType == CsObjectType.BBOX:
                obj = CsBbox()
            obj.fromJsonText(objIn, objId)
            self.objects.append(obj)

    def toJsonText(self):
        jsonDict = {}
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict = obj.toJsonText()
            jsonDict['objects'].append(objDict)

        return jsonDict

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())


# a dummy example
if __name__ == "__main__":
    obj = CsPoly()
    obj.label = 'car'
    obj.polygon.append( Point( 0 , 0 ) )
    obj.polygon.append( Point( 1 , 0 ) )
    obj.polygon.append( Point( 1 , 1 ) )
    obj.polygon.append( Point( 0 , 1 ) )

    print(type(obj).__name__)
    print(obj)
