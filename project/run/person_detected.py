from ultralytics.yolo.engine.model import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(source= '../video_test/oto_fire.mp4', show=False, stream=True, classes = 0, save = True)
for i, (result) in enumerate(results):
    print('Do something with class 0')