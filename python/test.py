import numpy as np
from stereo_vision import stereo_vision

sv = stereo_vision()

imgL = np.zeros((1242//4, 375//4,3), dtype=np.float32)
imgR = np.zeros((1242//4, 375//4,3), dtype=np.float32)

while True:
    sv.generatePointCloud(imgL, imgR)
