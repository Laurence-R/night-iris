import os
import shutil
from pathlib import Path

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set ROBOFLOW_API_KEY in the environment before running this script.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("object-detection-ma3wy").project("lod-dataset-rsqer")
version = project.version(2)
dataset = version.download("yolo26")

# Align Roboflow folder name with repo convention (enhanced / train set).
downloaded = Path(dataset.location)
target = Path("w_enhance")
if downloaded.resolve() != target.resolve():
    if target.exists():
        raise SystemExit(f"Target already exists: {target}. Remove or rename it before re-download.")
    shutil.move(str(downloaded), str(target))
    print(f"Renamed {downloaded.name} -> {target}")
else:
    print(f"Dataset ready at {target}")
