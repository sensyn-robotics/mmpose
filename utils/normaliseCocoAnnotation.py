import mmcv
from pathlib import Path
import pandas as pd


def rearrange(keypointDict, bboxDict):
    keypointDict['segmentation'] = bboxDict['segmentation']
    keypointDict['area'] = bboxDict['area']
    keypointDict['bbox'] = bboxDict['bbox']
    keypointDict['isbbox'] = bboxDict['isbbox']





cocoAnnotationSourcePath = Path.home()/"Downloads/meter-test-3.json"
annotationDict = mmcv.load(str(cocoAnnotationSourcePath))

count=0
annList = annotationDict['annotations']
newAnnList = []
for i in range(0,len(annList),2):
    print(annList[i]["image_id"])
    print(annList[i+1]["image_id"])
    print("----------------------------")
    assert annList[i]["image_id"]==annList[i+1]["image_id"]

    rearrange(keypointDict=annList[i], bboxDict=annList[i+1])
    #print(annList[i])
    newAnnList.append(annList[i])
    #print(annList[i])
    #break
annotationDict['annotations'] = newAnnList
print(len(annotationDict['annotations']))
#print(type(annotationDict))

#create one image annotationDict

#oneImageAnnotationDict = {
#    "images": annotationDict['images'][0],
#    "annotations": annotationDict['annotations'][0],
#    "categories": annotationDict['categories'][0],
#}


#oneImageAnnotationDict = {
#    "images": [annotationDict['images'][0]],
#    "annotations": [annotationDict['annotations'][0]],
#    "categories": [annotationDict['categories'][0]]
#}

#print(annotationDict['images'][1])
#twoImageAnnotationDict = {
#    "images": [annotationDict['images'][0], annotationDict['images'][1]],
#    "annotations": [annotationDict['annotations'][0], annotationDict['annotations'][1]],
#    "categories": [annotationDict['categories'][0], annotationDict['categories'][0]]
#}

#oneImageAnnotationDict = {}
#oneImageAnnotationDict['images'] = list(annotationDict['images'][0])
#oneImageAnnotationDict['annotations'] = list(annotationDict['annotations'][0])
#oneImageAnnotationDict['categories'] = list(annotationDict['categories'][0])


mmcv.dump(annotationDict, "newMeterAnn2.json")
#mmcv.dump(oneImageAnnotationDict, "newMeterAnnOneImageOnly.json")
#mmcv.dump(twoImageAnnotationDict, "newMeterAnnTwoImageOnly.json")


