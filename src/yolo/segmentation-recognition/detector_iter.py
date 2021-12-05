import cv2
import numpy as np
import sys
import time
import os

confThreshold = 0.5
nmsThreshold = 0.4

def getOutputLayers(net):
	layerNames = net.getLayerNames()
	outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return outputLayers

def drawBoundingBox(image, class_id, confidence, xmin, ymin, xmax, ymax):
	label = str(classes[class_id])
	cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
#	cv2.putText(image, label, (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.putText(image, label, (xmin-3, ymin-3), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)


# read class names from text file
classes = None
with open('crnet.names', 'r') as f:
	classes = [line.strip() for line in f.readlines()]

# read pre-trained model and config file
net = cv2.dnn.readNet('crnet_last.weights', 'crnet_last.cfg')
net2 = cv2.dnn.readNet('crnet_old.weights', 'crnet_old.cfg')

files = os.listdir("test/")

for file in files:
	class_ids = []
	confidences = []
	boxes = []
	image = cv2.imread("test/" + file)
	image2 = cv2.resize(image, (604, 220))
	width = image2.shape[1]
	height = image2.shape[0]
	blob = cv2.dnn.blobFromImage(image, 1/255, (352,128), (0,0,0), True, crop=False)
	net.setInput(blob)
	net2.setInput(blob)
	outs = net.forward(getOutputLayers(net))
	outs.extend(net2.forward(getOutputLayers(net2)))
	
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > confThreshold:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])

	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		drawBoundingBox(image2, class_ids[i], confidences[i], int(x), int(y), int(x+w), int(y+h))
	#image = cv2.resize(image, (604,220))
	cv2.imshow("detections", image2)
	cv2.waitKey(0)
