from ultralytics import YOLO

import cv2


personModel = YOLO('yolov8n.pt')
fireModel = YOLO('best.pt')

conf_fire = 0.1
conf_per = 0.7

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    ##handle with detected fire
    fireDetected =fireModel.predict(frame, show=False,conf = conf_fire)
        #get position each fire
    for results in fireDetected[0].boxes:
        # print(i[0])
        for box in results:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
            cv2.putText(frame, "Fire", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2,cv2.LINE_AA)

    ##handle with detecd person

    personDetected = personModel.predict(frame, classes = 0, show = False, conf = conf_per)
        #get position each person

    for results in personDetected[0].boxes:
        for box in results:

            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),5)
            cv2.putText(frame, "Person", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)

    

    cv2.imshow('NCKH 2023', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.relase()
cv2.destroyAllWindows()