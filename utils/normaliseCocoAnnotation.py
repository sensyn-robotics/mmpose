import mmcv
from pathlib import Path
import pandas as pd


def rearrange(keypointDict, bboxDict):
    keypointDict['segmentation'] = bboxDict['segmentation']
    keypointDict['area'] = bboxDict['area']
    keypointDict['bbox'] = bboxDict['bbox']
    keypointDict['isbbox'] = bboxDict['isbbox']





cocoAnnotationSourcePath = Path.home()/"Downloads/meter-test-2.json"
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
    print(annList[i])
    #break
annotationDict['annotations'] = newAnnList
print(len(annotationDict['annotations']))

mmcv.dump(annotationDict, "newMeterAnn.json")

