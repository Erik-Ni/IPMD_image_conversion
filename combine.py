#!/usr/bin/env python

# Copyright 2015 Google, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draws squares around detected faces in the given image."""

import os
import cv2
import pandas as pd
import numpy as np
from webimg import url_to_image

# [START import_client_library]
#from google.cloud import vision
# [END import_client_library]
#from google.cloud.vision import types

from landmarks import firstmodify, ifoverborder, finalmodify

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')


def deep_convert(f, pic_id, return_rectangle = False, save_img=False):
    image = url_to_image(f)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            len = detections[0, 0, i, 3:7]

            if len[3] < 1:
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(startX, 0)
                startY = max(startY, 0)
                #print(startX, startY)
                left, right, up, bottom = firstmodify(startX, endX, startY, endY)
                #print("1",left, right, up, bottom)
                left, right, up, bottom = ifoverborder(left, right, up, bottom, w, h)
                #print("2",left, right, up, bottom)
                left, right, up, bottom = finalmodify(left, right, up, bottom)
                #print("3",left, right, up, bottom)
                #print(left, right, up, bottom)
                roi = image[up:bottom, left:right]
                #roi = image[startY:endY, startX:endX]
                roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
                output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if save_img:
                    head, tail = os.path.split(f)
                    outfile = 'output/'+str(pic_id)+'-'+tail
                    cv2.imwrite(outfile, output)
                temp_output = output.flatten()
                if return_rectangle:
                    return temp_output, [left, right, up, bottom, w]
                return temp_output

