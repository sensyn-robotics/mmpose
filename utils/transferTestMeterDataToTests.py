from pathlib import Path
import shutil

sourceDir= Path.home()/"work/gaugeReading/images"
datadir = Path.home()/"work/coco-annotator/datasets/meter-test"
testFiles = [imF for imF in sourceDir.rglob("*.png")][-20:]
targetDir = Path.home()/"work/mmpose/tests/data/meter"

for imFile in testFiles:
    imName = imFile.name
    imPathInDataDir = datadir/imName
    if not imPathInDataDir.exists():
        print(imPathInDataDir)
        shutil.copy(imFile, targetDir)
    #break
