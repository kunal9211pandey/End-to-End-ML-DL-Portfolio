import cv2

# Correct class name
face_cas = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")

img = cv2.imread("couple.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(gray_img ,30, 200)

contours , hierarchy = cv2.findContours(edge ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE ) # return the tree structure of the contours
# hierarchy means tree of contours , adn retr_tree return structure of contours ,CHAIN_APPROX_SAMPLE resturn appropriate contours only


cv2.drawContours(img, contours, -1,(0,255,0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()