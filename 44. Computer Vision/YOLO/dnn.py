import cv2

net = cv2.dnn.readNet("yolov3.cfg")

classes = []
with open("coco.names" ,'r') as f:
    for line in f.readlines():
        classes.append(line.strip())
# print(classes)

print(len(classes))

