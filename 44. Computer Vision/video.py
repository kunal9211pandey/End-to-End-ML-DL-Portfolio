# video is nothing but group of frame ( frame is group of image)
import cv

video = cv2.VideoCapture(0) # 0 means capture video from  computer camera
# 1 means capture video from external cemera, also press the video to play it

video.release() # to close the camera

face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("dance1.mp4")

ret, frame = cap.read() # use this logic in loop beacuse video have lots of frame

gray_frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

face = face_cas.detectMultiScale(gray_frame, 1.3,5) # find the face inside the frame

for (x,y,w,h) in face:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('img',frame)
key = cv2.waitKey(1)
if key ==ord('q'):
    break