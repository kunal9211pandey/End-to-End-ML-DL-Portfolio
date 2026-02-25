import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# load coco object name file
classes = []
with open("coco.names", 'r') as f:
    for line in f.readlines():
        classes.append(line.strip())
print(classes)

# print(len(classes))

# load img
img = cv2.imread("test_img.jpg")

# resize the image (416 and 416)
img = cv2.resize(img, (416, 416))

# collecting hight and width and no. of chenel from image shape
height, width, channels = img.shape
print(height, width, channels)

# convert image to blob (binary large object)
# (0,0,0) scaler with mean RGB values which are subtracted from chanel
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)

# set input from yolo object detecation
net.setInput(blob)

# find name of all the layers
layers_name = net.getLayerNames()
print(layers_name)

# find the name of output layers
output_layers = []
for i in net.getUnconnectedOutLayers().flatten():
    output_layers.append(layers_name[i - 1])
print(output_layers)

# send blobdata to yolo model and send result to output layers
blobdata = net.forward(output_layers)

# blobdata is a group of 2d array that contain float number blobdata

# genratre random color for all 80 classes
colors = np.random.uniform(0, 255, size=(80, 3))

# detect the object in the image
class_ids = []
confidences = []
boxes = []

for x in blobdata:
    for detecation in x:
        # extract score value from 5th element onwards
        scores = detecation[5:]
        # ids of object with higest scores are stored
        # into class id
        class_id = np.argmax(scores)
        # confidence score for each object id
        confidence = scores[class_id]
        if confidence > 0.5:
            # extract values to draw rectangular box
            center_x = int(detecation[0] * width)
            center_y = int(detecation[1] * height)
            w = int(detecation[2] * width)
            h = int(detecation[3] * height)
            # rectangle cooridinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            # measurement of rectangular boxes are stored into boxes
            boxes.append([x, y, w, h])
            # accureecy scores are stored inot confidence
            confidences.append(float(confidence))
            # class id s are stored into class_ids
            class_ids.append(class_id)

# elinates multiple boxes to be display around the image
# this is called non maximum su[[ression of boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# draw rectangular box with text for each object
# repeat for each box
for i in indexes.flatten():
    # extract each box measuremnt
    x, y, w, h = boxes[i]
    # extract the label based on the class id of box
    label = classes[class_ids[i]]
    # confidence in precentage
    confidence_score = int(confidences[i] * 100)
    # extract color based on i
    color = colors[class_ids[i] % 80]
    # draw the rectangel around the image
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # construct string in the rectangluare box
    str1 = '%s %d%%' % (label, confidence_score)
    # display the string in the recatunglure boxes
    cv2.putText(img, str1, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()