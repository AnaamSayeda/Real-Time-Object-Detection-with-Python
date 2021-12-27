import cv2
import numpy as np

vid = cv2.VideoCapture(0)
vid.set(3,800)
vid.set(4,700)
vid.set(10,70)
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


while True:
    success, img = vid.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('output', img)
    cv2.waitKey(0)
