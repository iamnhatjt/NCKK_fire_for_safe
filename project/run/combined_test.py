from ultralytics import YOLO

import cv2
import time
import threading


personModel = YOLO('yolov8n.pt')
fireModel = YOLO('best.pt')

#status
status = ['','canhbao', 'nguy co', 'nguy hiem']

# set fps
fps = 2.5 # kiểm tra bao nhiêu khung hình trên một giây < lấy gần bằng>

conf_fire = 0.3 # hệ số cài đặt nhận diện lửa
conf_per = 0.6 # hệ số cài đặt nhân diện con người


frame_count = 0 # số frame để tính fps
start_time = time.time() # thời gian chương trình bắt đầu chạy < lấy đây là gốc tương đối so sánh các mốc thời gian khác, có thể xóa nếu vào chương trình thật>
time_notication = 0


#rate fire and person
rate_fire = False # dùng để kiểm tra điều kiện hiển thị thông báo
rate_person = False 

#count_detected fire and person
count_frame_fire = 0 #số frame detect lửa, sẽ set về 0 lại trong nhiều trường hợp   
count_frame_person = 0 # số frame detect con người, sẽ set lại về 0 trong nhiều trường hợp

#time_reset_all
global start_time_fire # thời gi bắt đầu phát hiện ngọn lửa đầu tiên
start_time_fire = 0
start_time_person = 0

#mediate_setting
mediate_fire = False #các biến trung gian, là nguồn dẫn điều kiện để thực hiện các điều kiện khác trong vòng lạp while
mediate_notication = False
mediate_person = False
canh_bao = False

#time setting
time_fire_danger = 10 #thoi gian gui canh bao dau tien
time_fire_danger2 = 10

cap = cv2.VideoCapture('../video_test/oto.mp4')
# cap = cv2.VideoCapture(0)

def wait_10s():
    time.sleep(time_fire_danger2)
    global canh_bao
    canh_bao = True



while True:
    ret, frame = cap.read()
    frame_count += 1 # dùng để tính fps

    #test code
    print("-----------mediate_fire", mediate_fire)
    print("-----------time run realtime: ",time.time() - start_time)
    ##handle with detected fire
    fireDetected =fireModel.predict(frame, show=False,conf = conf_fire) #model phát hiện lửa

        #get position each fire
    for results in fireDetected[0].boxes: #xử lý trong những ngọn lửa

        ##start handle with detectfire = true
        if(mediate_fire == False):
            
            if(count_frame_fire != 0):
                mediate_person = True
                mediate_fire = True
                start_time_fire = time.time()
                count_frame_person = 0
                print('--------Time check fire start counting!')
            
        


        if(mediate_fire == True):
            print('-----------------time check fire: ', time.time() - start_time_fire )
            print("-----------------------rate_fire: ",count_frame_fire/ ((time.time() -start_time_fire + 1) * fps)  )

        
        
        count_frame_fire += 1/ len(fireDetected[0].boxes)
        print(count_frame_fire)
        print(count_frame_person)

        


        # print(i[0])
        for box in results:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
            cv2.putText(frame, "Fire", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2,cv2.LINE_AA)
            

    if(time.time() - start_time_fire > time_fire_danger and time.time() - start_time_fire < time_fire_danger + 10 ):
            print("-----------------------rate_fire: ",count_frame_fire/ ((time.time() -start_time_fire + 1) * fps)  )
            if(count_frame_fire/ ((time.time() -start_time_fire + 1) * fps)  > 0.6):
                print(' gui loi canh bao toi nguoi dung ------------------------------')
                rate_fire = True
            mediate_fire = False
            count_frame_fire = 0
            if(count_frame_person/ ((time.time() -start_time_fire + 1) * fps)  > 0.6):
                rate_person = True
            count_frame_person = 0
            start_time_fire = 0

            

            
    
    if(rate_fire):

        if(rate_person == False):
            if canh_bao == True:
                cv2.putText(frame, 'Nguy hiem', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            else:
                cv2.putText(frame, 'Canh bao', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                t = threading.Thread(target=wait_10s)

                t.start()
            
        else:
            cv2.putText(frame, 'can than', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
        if(mediate_notication == False):
                time_notication = time.time()
                mediate_notication = True
                print('-----------------------------------------------------------------')
        elif(time.time() - time_notication > 5 and time.time() - time_notication < 10):
            rate_fire = False
            mediate_notication = False



    print('rate_person: ', rate_person)
    print('person frame: ', count_frame_person)





    ##handle with detecd person

    personDetected = personModel.predict(frame, classes = 0, show = False, conf = conf_per)
        #get position each person

    for results in personDetected[0].boxes:
    
                    
        count_frame_person += 1/ len(personDetected[0].boxes)     
             



        for box in results:
                      


            #start handle person with detectperson = True
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),5)
            cv2.putText(frame, "Person", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),2,cv2.LINE_AA)

    

    

    fps = frame_count/ (time.time() - start_time)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('NCKH 2023', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

