import cv2
import numpy as np

img = cv2.imread('img.png')
classNames = []

file = 'coco.names'
with open(file, 'rt') as r:
    classNames = r.read().rstrip('\n').split('\n')

configuration = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights, configuration)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classid, confidence, bbox = net.detect(img, confThreshold=0.5)
print(classid, bbox)

for id, conf, box in zip(classid.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, box, color=(255, 255, 0), thickness=5)
    cv2.putText(img, classNames[id - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 0), 2)

cv2.imshow('output', img)
cv2.waitKey(0)
