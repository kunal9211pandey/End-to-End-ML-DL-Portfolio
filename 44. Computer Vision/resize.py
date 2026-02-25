import cv2

img = cv2.imread("taj1.jpg")

print(img.shape)

"""
resize_image = cv2.resize(img, (400, 700))
cv2.imshow("at the taj mahal" , resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# resize image mannulay

h = int(img.shape[0]/3)
w = int(img.shape[1]/3)

resize_image = cv2.resize(img ,(w, h))
cv2.imshow("at the taj mahal" , resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
