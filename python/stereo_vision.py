import ctypes
import os
import sys
import cv2
from numpy.ctypeslib import ndpointer

class stereo_vision:

    def __init__(self, so_lib_path='bin/stereo_vision.so'):
        self.sv = ctypes.CDLL(so_lib_path)
        self.sv.generatePointCloud.restype = ndpointer(dtype=ctypes.c_double, shape=(1242*375,3))

    def generatePointCloud(self, left, right):
        left = cv2.cvtColor(left, cv2.COLOR_BGR2BGRA)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2BGRA)
        length = left.shape[0]
        width = left.shape[1]
        left = left.tostring()
        right = right.tostring()
        return self.sv.generatePointCloud(left, right, width, length)

if __name__ == "__main__":
    s = stereo_vision()
    kittiPath = sys.argv[1]
    scale_factor = 4
    for iFrame in range(465):
        leftName  = "{}/video/testing/image_02/0000/{:0>6}.png".format(kittiPath, iFrame)
        rightName = "{}/video/testing/image_03/0000/{:0>6}.png".format(kittiPath, iFrame)
        #print(leftName, rightName)
        left = cv2.imread(leftName)
        right = cv2.imread(rightName)
        left = cv2.resize(left, (left.shape[1]//scale_factor, left.shape[0]//scale_factor))
        right = cv2.resize(right, (right.shape[1]//scale_factor, right.shape[0]//scale_factor))
        s.generatePointCloud(left, right)
    