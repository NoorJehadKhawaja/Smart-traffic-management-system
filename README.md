# Smart-traffic-management-system
using custom dataset for object detection using YOLO model optimizes traffic flow, prioritizes emergency vehicles, and enhances safety at intersections.

we use google colab to run our code
!nvidia-smi
import os
HOME = os.getcwd()
print(HOME)
Pip install method (recommended)

!pip install ultralytics==8.0.196

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
Git clone method (for development)

%cd {HOME}
!git clone github.com/ultralytics/ultralytics
%cd {HOME}/ultralytics
!pip install -e .

 from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from IPython.display import display, Image
!mkdir {HOME}/datasets
%cd {HOME}/datasets

# we use aroboflow website to make our dataset with annotaion

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="sdxIR5efXXEFYsED0Fde")
project = rf.workspace("noor-7vdrq").project("smart-traffic-management-system-xne3g")
version = project.version(5)
dataset = version.download("yolov8")
%cd {HOME}

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=100 imgsz=800 plots=True

!ls {HOME}/runs/detect/train/

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True

