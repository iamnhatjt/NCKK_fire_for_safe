from ultralytics.yolo.engine.model import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(source="0", show=True, stream=True, classes = 0)
for i, (result) in enumerate(results):
    print('Do something with class 0')
    print(result[0].boxes[0])