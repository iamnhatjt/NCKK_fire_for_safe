from ultralytics import YOLO

# model = YOLO("../model/best.pt") # depen on pwd "WATCHOUT"
model = YOLO('best.pt')
results = model.predict(show=True, source=0)
print(results)

# results = model.predict(show=True, source="0")
# print(results)




#save video after detect
#yolo task=detect mode=predict model=<path to weight file> conf=0.25 source=<path to source image or  video> save=True