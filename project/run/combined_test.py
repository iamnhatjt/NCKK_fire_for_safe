from ultralytics import YOLO

import cv2
import threading

#import model was train
personModel = YOLO('yolov8n.pt')
fireModel = YOLO('best.pt')


cap = cv2.VideoCapture(0)
#function handle fire detection

    


#function handle personDetection
def detectionPersion(frame):
    
    personDetected = personModel.predict(frame, classes = 0, show = False)
        #get position each person
    for results in personDetected[0].boxes:
        for box in results:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),5)
            cv2.putText(frame, "Person", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)

def detection_fire(frame):
    fireDetected =fireModel.predict(frame, show=False, device=False)
        #get position each fire
    for results in fireDetected[0].boxes:
        for box in results:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
            cv2.putText(frame, "Fire", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2,cv2.LINE_AA)

while True:

    ret, frame = cap.read()


    # tạo và bắt đầu thực thi tiến trình xử lý phát hiện người
    detectingPerson = threading.Thread(target=detectionPersion, args=(frame,))
    detecting_fire = threading.Thread(target=detection_fire, args=(frame,))

    detectingPerson.start()
    detecting_fire.start()

    detectingPerson.join()
    detecting_fire.join()

    
    
    

   



    

    cv2.imshow('NCKH 2023', frame)
 
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()