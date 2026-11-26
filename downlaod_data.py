

from roboflow import Roboflow
rf = Roboflow(api_key="Key")
project = rf.workspace("prueba-bl8vt").project("oil-spill-sfzvs")
version = project.version(6)
dataset = version.download("yolov8", location="data/oil_spill")
                 