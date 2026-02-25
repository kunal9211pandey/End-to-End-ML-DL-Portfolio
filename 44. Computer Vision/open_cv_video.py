import cv2   # fixed import

face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("dance1.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:   # stop when video ends
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection added (missing in your code)
    face = face_cas.detectMultiScale(gray_frame, 1.1, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()