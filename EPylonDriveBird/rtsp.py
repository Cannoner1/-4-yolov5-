import cv2

url="rtsp://admin:admin123@192.168.10.50:554/cam/realmonitor?channel=1&subtype=0"
#rtsp://admin:quniao02@192.168.10.50:554/ch1/sub/av_stream
#sudo python3 /home/cat/camana/rtsp.py

cap=cv2.VideoCapture(url)

if cap.isOpened():
    rval,frame=cap.read()
else:
    print("error")
    rval=False

while rval:
    frame=cv2.resize(frame,(640,480))
    cv2.imshow("cam_numl",frame)
    rval,frame=cap.read()
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()

cv2.destroyAllWindows()