import os

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set ROBOFLOW_API_KEY in the environment before running this script.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("object-detection-ma3wy").project("lod-dataset-rsqer")
version = project.version(2)
dataset = version.download("yolo26")
