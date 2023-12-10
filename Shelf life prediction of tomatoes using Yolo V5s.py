# -*- coding: utf-8 -*-
"""TomatoObj_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16bTppdY9yYTHX3m2bMKt6bMk6env8bzc
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # clone
# %cd yolov5
# %pip install -qr requirements.txt comet_ml  # install

import torch
import utils
display = utils.notebook_init()  # checks

# Commented out IPython magic to ensure Python compatibility.
#@title Select YOLOv5 🚀 logger {run: 'auto'}
logger = 'Comet' #@param ['Comet', 'ClearML', 'TensorBoard']

if logger == 'Comet':
#   %pip install -q comet_ml
  import comet_ml; comet_ml.init()
elif logger == 'ClearML':
#   %pip install -q clearml
  import clearml; clearml.browser_login()
elif logger == 'TensorBoard':
#   %load_ext tensorboard
#   %tensorboard --logdir runs/train

!unzip -q ../train_data.zip -d ../

!python train.py --img 640 --batch 10 --epochs 81 --data custom_data.yaml --weights yolov5s.pt --cache

!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../innn.jpg   --save-txt

pip install torchsummary

import torch
from torchsummary import summary
from models.yolo import Model

# Load YOLOv5 model
model = Model('/content/yolov5/data/custom_data.yaml')


# Specify device (cuda or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model.to(device)

# Print the model summary
summary(model, input_size=(3, 640, 640), device=device)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
model.eval()