import cv2

# Correct class name
face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Image name must be in quotes
img = cv2.imread("delhi2.jpg")

# Check if image loaded
if img is None:
    print("Image not found. Check file name or path.")
else:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cas.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
    print(faces)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()