
def test_demo():
	import stereo_vision
	import numpy as np
	width = 1242
	height = 375
	sv = stereo_vision.stereo_vision(objectTracking=False, width=width, height=height)
	# /home/aditya/.local/lib/python3.9/site-packages/stereo_vision_serial.cpython-39-x86_64-linux-gnu.so
	l = np.zeros((width, height), dtype=np.uint8)
	r = np.zeros((width, height), dtype=np.uint8)
	sv.generatePointCloud(l, r)


if __name__=="__main__":
	test_demo()