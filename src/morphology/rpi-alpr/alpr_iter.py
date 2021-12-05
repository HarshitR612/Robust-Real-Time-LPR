import cv2
import numpy as np
from skimage.filters import threshold_local
from imutils import resize
import json
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
from keras.models import model_from_json
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import time

def _get_available_gpus():  
	return []
tfback._get_available_gpus = _get_available_gpus

# constants
SEL = cv2.getStructuringElement(cv2.MORPH_RECT, (17,4))
kernel = np.ones((3,3), np.uint8)


# load models
fd = open('lenet5char.json', 'r')
char_json = fd.read()
fd.close()
char_recog = model_from_json(char_json)
char_recog.load_weights('lenet5char.h5')
fd = open('lenet5digit.json', 'r')
digit_json = fd.read()
fd.close()
digit_recog = model_from_json(digit_json)
digit_recog.load_weights('lenet5digit.h5')

def custom_sort(r):
	return r[0]

format = sys.argv[1]
path = "cars/" + sys.argv[1] + "/"
files = os.listdir(path)

comp_time = 0
f = open(sys.argv[1] + ".txt", "w")
for file in files:
	license_plate_str = ""
	image = cv2.imread(path + file)
	image = cv2.resize(image, (260,175))
	image = image[68:175, 50:220]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	start = time.time()
	gray = cv2.medianBlur(gray, 3)
	modif = cv2.blur(gray, (20,20))
	gray = cv2.subtract(gray, modif)
	gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
	gray = cv2.erode(gray, kernel, iterations=1)
	gray = cv2.dilate(gray, kernel, iterations=1)
	contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	for cnt in contours:
		character_images = []
		rect = cv2.minAreaRect(cnt)
		box = np.int0(cv2.boxPoints(rect))
		angle = rect[2]
		width, height = rect[1]
		#angle = (angle + 180) if width < height else (angle + 90)
		xmin, ymin = np.amin(box, axis=0)
		xmax, ymax = np.amax(box, axis=0)
		area = (xmax - xmin) * (ymax - ymin)
		if height*width <= 0:
			continue
		coords = cv2.boundingRect(cnt)
		box[:,0] = box[:,0] - xmin
		box[:,1] = box[:,1] - ymin
		M = np.zeros((coords[3], coords[2]), np.uint8)
		cv2.fillConvexPoly(M, box, 255)
		fig = gray[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
		density = np.count_nonzero(np.bitwise_and(fig, M)) / area if area > 0 else 0
		if width == 0 or height == 0:
			aspr = 0
		else:
			aspr = width/height if width > height else height/width
		if density >= 0.5 and density <= 0.95 and aspr >= 3 and aspr <= 10 and area >= 450 and area <= 35000:
			minx, miny, maxx, maxy = coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]
			plate = image[miny:maxy, minx+2:maxx+2]
			#blurred = cv2.GaussianBlur(plate, (5,5), 1.0)
			#sharpened = float(3) * plate - float(2) * blurred
			#sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
			#sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
			#sharpened = sharpened.round().astype(np.uint8)
			V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
			T = threshold_local(V, 29, offset=15, method="gaussian")
			thresh = (V > T).astype("uint8") * 255
			thresh = cv2.bitwise_not(thresh)
			thresh = resize(thresh, width=400)
			char_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
			char_coords = [cv2.boundingRect(char_cnt) for char_cnt in char_contours]
			char_coords = sorted(char_coords, key=custom_sort)
			minht, maxht = 0.4*thresh.shape[0], 0.8*thresh.shape[0]
			license_plate_str = ''
			i = 0
			for cc in char_coords:
				wd, ht = cc[2], cc[3]
				char_roi = thresh[cc[1]:cc[1]+ht, cc[0]:cc[0]+wd]
				char_area = wd*ht
				char_density = np.count_nonzero(char_roi)/char_area if char_area > 0 else 0
				if ht == 0 or wd == 0:
					char_aspr = 0
				else:
					char_aspr = wd/ht if wd>ht else ht/wd
				if ht >= minht and wd >= 10 and wd <= 50 and char_density >= 0.36 and char_aspr >= 1.49:
					img = cv2.resize(char_roi, (28,28))
					img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
					img = img.flatten()
					img = img.reshape((1,784))
					character_images.append(img)
			if format == '5d10':
				try:
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[0])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[1])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[2])[0]))
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[3])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[4])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[5])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[6])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[7])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[8])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[9])[0]))
				except IndexError:
					pass
			elif format == '6d10':
				try:
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[0])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[1])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[2])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[3])[0]))
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[4])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[5])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[6])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[7])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[8])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[9])[0]))
				except IndexError:
					pass
			elif format == '5d9':
				try:
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[0])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[1])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[2])[0]))
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[3])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[4])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[5])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[6])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[7])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[8])[0]))
				except IndexError:
					pass
			elif format == '6d9':
				try:
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[0])[0]) + 65)
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[1])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[2])[0]))
					license_plate_str += chr(np.argmax(digit_recog.predict(character_images[3])[0]))
					license_plate_str += chr(np.argmax(char_recog.predict(character_images[4])[0]) + 65)
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[5])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[6])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[7])[0]))
					license_plate_str += str(np.argmax(digit_recog.predict(character_images[8])[0]))
				except:
					pass
	comp_time += (time.time() - start)	
	f.write(license_plate_str + "\n")
f.close()
if len(files) > 0:
	avg = (comp_time*1000)/len(files)
	print("Average Execution Time : " + str(avg) + "ms")
