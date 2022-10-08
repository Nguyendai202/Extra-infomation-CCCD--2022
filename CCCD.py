import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision
import time
import vietocr
import PIL
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pyodbc
import fastapi
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
from typing import List


sys.path.append("D:\\Projects\\CCCD")
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

source_image_path = "C:\\Users\\scien\\Downloads\\img083.jpg"
image = cv2.imread(source_image_path)

classes_to_filter_corner = ['tl', 'tr', 'bl', 'br'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt_corner  = {
    
    "weights": "D:\\Projects\\CCCD\\weights\\best-corner-detection.pt", # "weights/yolov7.pt", # "weights/best-text-detection.pt"# Path to weights file default weights are for nano model
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter_corner  # list of classes to filter or None

}
classes_to_filter_text = ['id', 'text'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt_text  = {
    
    "weights": "D:\\Projects\\CCCD\\weights\\best-text-detection-3.pt", # "weights/yolov7.pt", # "weights/best-text-detection.pt"# Path to weights file default weights are for nano model
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter_text  # list of classes to filter or None

}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def box_iou(box1, box2):
  def box_area(box):
      # box = 4xn
      return (box[2] - box[0]) * (box[3] - box[1])

  area1 = box_area(box1.T)
  area2 = box_area(box2.T)

  # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
  inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
  return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

img0 = image.copy()
output_boxes_corner = []
nb = 0
start = time.time()
with torch.no_grad():
  weights, imgsz = opt_corner['weights'], opt_corner['img-size']
  set_logging()
  device = select_device(opt_corner['device'])
  half = device.type != 'cpu'
  model = attempt_load(weights, map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(imgsz, s=stride)  # check img_size
  if half:
    model.half()

  names = model.module.names if hasattr(model, 'module') else model.names
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

  img = letterbox(img0, imgsz, stride=stride)[0]
  img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
  img = np.ascontiguousarray(img)
  img = torch.from_numpy(img).to(device)
  img = img.half() if half else img.float()  # uint8 to fp16/32
  img /= 255.0  # 0 - 255 to 0.0 - 1.0
  if img.ndimension() == 3:
    img = img.unsqueeze(0)

  # Inference
  t1 = time_synchronized()
  pred = model(img, augment= False)[0]
  # Apply NMS
  classes = None
  if opt_corner['classes']:
    classes = []
    for class_name in opt_corner['classes']:

      classes.append(names.index(class_name))

  if classes:
    
    classes = [i for i in range(len(names)) if i not in classes]
  
#########################
  conf_thres = opt_corner['conf-thres']
  iou_thres = opt_corner['iou-thres']
  labels = ()
  multi_label = False
  agnostic = False

  # NMS - checking..................
  nc = pred.shape[2] - 5  # number of classes
  xc = pred[..., 4] > conf_thres  # candidates
    # Settings
  min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
  max_det = 300  # maximum number of detections per image
  max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
  time_limit = 10.0  # seconds to quit after
  redundant = True  # require redundant detections
  multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
  merge = False  # use merge-NMS
  t = time.time()
  output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]
  for xi, x in enumerate(pred):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height  # confidence
        x = x[xc[xi]]
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
# #########################
  pred = output
  t2 = time_synchronized()
  for i, det in enumerate(pred):
    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
    if len(det):
      det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

      for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
      for *xyxy, conf, cls in reversed(det):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        output_boxes_corner.append(xywh)
        nb += 1
        # label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img0, label= None, color=colors[int(cls)], line_thickness=3)
end = time.time()
elapsed_time = end - start
print ("elapsed_time:{0}".format(elapsed_time) + "s")
print(output_boxes_corner)
print(nb)

crop_list = []
for box in output_boxes_corner:
  center_x = int(float(box[0]))
  center_y = int(float(box[1]))
  crop_list.append([center_x, center_y])
crop_x = []
crop_y = []
for i in range(nb):
  crop_x.append(crop_list[i][0])
  crop_y.append(crop_list[i][1])
x = min(crop_x)
w = max(crop_x) - x
y = min(crop_y)
h = max(crop_y) - y
crop_image = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
  #############
frame_image = np.float32([[0, 0], [500, 0], [500, 400], [0, 400]])
matrix = cv2.getPerspectiveTransform(crop_image, frame_image)
cropped = cv2.warpPerspective(image, matrix, (500, 400))
  ################
image = cropped

img1 = image.copy()
output_boxes_text = []
class_text = []
nb2 = 0
start2 = time.time()
with torch.no_grad():
  weights2, imgsz2 = opt_text['weights'], opt_text['img-size']
  set_logging()
  device2 = select_device(opt_text['device'])
  half2 = device2.type != 'cpu'
  model2 = attempt_load(weights2, map_location=device2)  # load FP32 model
  stride2 = int(model2.stride.max())  # model stride
  imgsz2 = check_img_size(imgsz2, s=stride2)  # check img_size
  if half2:
    model2.half()

  names2 = model2.module.names if hasattr(model2, 'module') else model2.names
  colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in names2]
  if device2.type != 'cpu':
    model2(torch.zeros(1, 3, imgsz2, imgsz2).to(device2).type_as(next(model2.parameters())))

  img2 = letterbox(img1, imgsz2, stride=stride2)[0]
  img2 = img2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
  img2 = np.ascontiguousarray(img2)
  img2 = torch.from_numpy(img2).to(device2)
  img2 = img2.half() if half2 else img2.float()  # uint8 to fp16/32
  img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
  if img2.ndimension() == 3:
    img2 = img2.unsqueeze(0)

  # Inference
  t12 = time_synchronized()
  pred2 = model2(img2, augment= False)[0]
  # Apply NMS
  classes2 = None
  if opt_text['classes']:
    classes2 = []
    for class_name in opt_text['classes']:

      classes2.append(names2.index(class_name))

  if classes2:
    
    classes2 = [i for i in range(len(names2)) if i not in classes2]
  
#########################
  conf_thres2 = opt_text['conf-thres']
  iou_thres2 = opt_text['iou-thres']
  labels2 = ()
  multi_label2 = False
  agnostic2 = False

  # NMS - checking..................
  nc2 = pred2.shape[2] - 5  # number of classes
  xc2 = pred2[..., 4] > conf_thres2  # candidates
    # Settings
  min_wh2, max_wh2 = 2, 4096  # (pixels) minimum and maximum box width and height
  max_det2 = 300  # maximum number of detections per image
  max_nms2 = 30000  # maximum number of boxes into torchvision.ops.nms()
  time_limit2 = 10.0  # seconds to quit after
  redundant2 = True  # require redundant detections
  multi_label2 &= nc2 > 1  # multiple labels per box (adds 0.5ms/img)
  merge2 = False  # use merge-NMS
  output2 = [torch.zeros((0, 6), device=pred2.device)] * pred2.shape[0]
  for xi, x in enumerate(pred2):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height  # confidence
        x = x[xc2[xi]]
        # Cat apriori labels if autolabelling
        if labels2 and len(labels2[xi]):
            l = labels2[xi]
            v = torch.zeros((len(l), nc2 + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc2 == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label2:
            i, j = (x[:, 5:] > conf_thres2).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres2]

        # Filter by class

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms2:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms2]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic2 else max_wh2)  # classes
        boxes2, scores2 = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes2, scores2, iou_thres2)  # NMS
        if i.shape[0] > max_det2:  # limit detections
            i = i[:max_det2]
        if merge2 and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes2[i], boxes2) > iou_thres2  # iou matrix
            weights = iou * scores2[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant2:
                i = i[iou.sum(1) > 1]  # require redundancy

        output2[xi] = x[i]
# #########################
  res = output2
  t2 = time_synchronized()
  for i, det in enumerate(res):
    s2 = ''
    s2 += '%gx%g ' % img2.shape[2:]  # print string
    gn2 = torch.tensor(img1.shape)[[1, 0, 1, 0]]
    if len(det):
      det[:, :4] = scale_coords(img2.shape[2:], det[:, :4], img1.shape).round()

      for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s2 += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "  # add to string
    
      for *xyxy, conf, cls in reversed(det):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        output_boxes_text.append(xywh)
        class_text.append(names2[int(cls)])
        nb2 += 1
        # label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img1, label= None, color=colors2[int(cls)], line_thickness=3)
end2 = time.time()
elapsed_time2 = end2 - start2
print ("elapsed_time:{0}".format(elapsed_time2) + "s")

info_crop = []
save_path = "D:\\Projects\\CCCD\\save"
index = 0
for box in output_boxes_text:
  center_x = int(float(box[0]))
  center_y = int(float(box[1]))
  w = int(float(box[2]))
  h = int(float(box[3]))
  crop_image = image[center_y - h // 2 : center_y + h // 2, center_x - w // 2 : center_x + w // 2]
  #############
  piece_path = save_path + "\\piece" + str(index) + ".jpg"
  cv2.imwrite(piece_path, crop_image)
  index += 1
  ################

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = "D:\\Projects\\CCCD\\transformerocr.pth"
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)

content_list = []
i_content = 0
position_y = []
position_x = []
for i in os.listdir(save_path):
  chr_img = Image.open(os.path.join(save_path, i))
  content = detector.predict(chr_img)
  content_list.append(content)
  position_x.append(output_boxes_text[i_content][0])
  position_y.append(output_boxes_text[i_content][1])
  os.remove(os.path.join(save_path, i))
  i_content += 1
print(content_list)
height = image.shape[0]

sorted_y = sorted(position_y)
rates = []
for element in sorted_y:
  indx = position_y.index(element)
  rate = float(position_y[indx] / height)
  rates.append(rate)
  print(content_list[indx])
  print("------------")

info = []

fields = ['ID', 'Họ và tên', 'Ngày tháng năm sinh', 'Giới tính', 'Quốc tịch', 'Quê quán', 'Nơi thường trú', 'Dân tộc']
#####################
# In ra ID, Họ tên và Ngày tháng năm sinh
for element in sorted_y[:3]:
  indx = position_y.index(element)
  print(fields[sorted_y.index(element)])
  print(content_list[indx])
  info.append(content_list[indx])
  print("-------------")
######################
# Xử lý và in ra giới tính, quốc tịch
a = position_y.index(sorted_y[3])
b = position_y.index(sorted_y[4])
sex = 0
nationality = 0
if position_x[a] < position_x[b]:
  sex = a
  nationality = b
else:
  sex = b
  nationality = a
print(fields[3])
print(content_list[sex])
info.append(content_list[sex])
print("-------------")
ctn = content_list[nationality]
if ctn == 'Việt Nam':
  print(fields[4])
else:
  print(fields[7])
print(ctn)
info.append(ctn)
print("-------------")
#######################
# Xử lý và in ra quê quán, nơi thường trú
home = sorted_y[5 : len(sorted_y)]
home_1 = []
home_2 = []
for i in home:
  rate = float(i / height)
  if rate >= 0.87875 - (1 - max(rates)):
    home_2.append(i)
  else:
    home_1.append(i)
print(fields[5])
string_1 = ''
for i in home_1:
  indx = position_y.index(i)
  print(content_list[indx])
  string_1 += content_list[indx]
print("-------------")
string_2 = ''
print(fields[6])
for i in home_2:
  indx = position_y.index(i)
  print(content_list[indx])
  string_2 += content_list[indx]
print("-------------")
info.append(string_1)
info.append(string_2)
print(1.1415926)
print(info)

personal_number_0 = info[0]
name_0 = info[1]
birth_0 = info[2]
sex_0 = info[3]
nationality_0 = info[4]
home_0 = info[5]
address_0 = info[6]

connection_string = 'mssql+pyodbc://LHP-3205/LHP-3205?driver=ODBC+Driver+17+for+SQL+Server'
engine = create_engine(connection_string)
Session = sessionmaker(bind= engine)
session = Session()
Base = declarative_base()

class Person(Base):
    __tablename__ = 'Information'

    id = Column(Integer, primary_key= True)
    personal_number = Column(String(20))
    name = Column(String(50))
    birth = Column(String(20))
    sex = Column(String(10))
    nationality = Column(String(10))
    home = Column(String(250))
    address = Column(String(250))
    
# Base.metadata.create_all(engine)
person = Person(personal_number= personal_number_0, name= name_0, birth= birth_0, sex= sex_0, nationality= nationality_0, home= home_0, address= address_0)
session.add(person)
session.commit()

app = FastAPI()
class PersonApi(BaseModel):
    id: int
    personal_number: str
    name: str
    birth: str
    sex: str
    nationality: str
    home: str
    address: str 
list_info = []
@app.get('/')
async def home():
    return 'Welcome!'
@app.post('/detect')
async def detect(file: UploadFile = File()):
    pass

@app.get('/show') 
async def show():
    return list_info
@app.get('/find/{id}')
async def find(id: int):
    try:
        return list_info[id]
    except:
        return HTTPException(status_code= 404, detail= 'Not Found')