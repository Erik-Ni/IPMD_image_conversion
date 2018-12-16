import numpy as np
import urllib.request
import cv2

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()))
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image