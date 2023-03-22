from ultralytics import YOLO

model = YOLO("../model/pretrain/yolo/yolov8n.pt") # pass any model type
model.train( data = '../dataset/mydataset.yaml',epochs=50)