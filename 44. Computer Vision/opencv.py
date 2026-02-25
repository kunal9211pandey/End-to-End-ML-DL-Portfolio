import cv2
img = cv2.imread("taj1.jpg")
# print(img)

# print(img.shape)

cv2.imshow("at taj mahal", img)
cv2.waitKey(0)  # # presh 0 to end the program
cv2.destroyAllWindows()  # close the program and all the window
