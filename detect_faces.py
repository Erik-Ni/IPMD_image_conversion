# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import glob
import re
import pandas as pd
import pickle

filepath = input("Filepath: ")
os.mkdir('output')
error_file = ''
number_of_error = 0
output_array = []

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

for f in glob.glob(filepath + '/**/*.*', recursive=True):

	head, tail = os.path.split(f)
	try:
		photo_id = re.findall(r'\d+', tail)[0]
		emotion = re.findall(r' ([A-Za-z]+)\.', tail)[0]
	except IndexError:
		print(tail + ' error')

	find = False
	image = cv2.imread(f)

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			len = detections[0, 0, i, 3:7]
			if len[3] < 1:
				find = True

				(startX, startY, endX, endY) = box.astype("int")
				startX = max(startX, 0)
				startY = max(startY, 0)

				roi = image[startY:endY, startX:endX]
				cv2.imshow("cropped", roi)
				roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
				output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

				temp_output = output.flatten()
				output_array.append([photo_id, temp_output, emotion.strip().title(), tail])

				# show the output image
				outfile = 'output/' + tail
				cv2.imwrite(outfile, output)

	if not find:
		error_file += f + ': fail to find the front face\n'
		number_of_error+=1

error_file += str(number_of_error) + ' errors\n'
output_pd = pd.DataFrame(output_array, columns =['id', 'pixels', 'emotion', 'original_file'])

with open('pixel.pd', 'wb') as fout:
	pickle.dump(output_pd, fout)
with open('error.txt', 'w') as fout:
	fout.write(error_file)
