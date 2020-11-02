# import the necessary packages
import numpy as np
import urllib
import urllib.request
import cv2
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image


while True:
	frame = url_to_image("http://192.168.0.109/capture")
	cv2.imshow('frame', frame)
	
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
