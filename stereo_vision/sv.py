import ctypes
import os
import sys
import cv2
from numpy.ctypeslib import ndpointer

class stereo_vision:

    def __init__(self, so_lib_path='bin/stereo_vision.so', width=1242, height=375, 
                defaultCalibFile=True, objectTracking=False, graphics=False, display=False, scale=1, pc_extrapolation=1):
        self.sv = ctypes.CDLL(so_lib_path)
        self.width = width
        self.height = height
        self.sv.generatePointCloud.restype = ndpointer(dtype=ctypes.c_double, shape=(width*height,3))
        
        self.defaultCalibFile = defaultCalibFile
        self.objectTracking = objectTracking
        self.graphics = graphics
        self.display = display
        self.scale = scale
        self.pc_extrapolation = pc_extrapolation

    def generatePointCloud(self, left, right):
        left = cv2.cvtColor(left, cv2.COLOR_BGR2BGRA)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2BGRA)
        left = left.tostring()
        right = right.tostring()
        return self.sv.generatePointCloud(left, right, self.width, self.height, self.defaultCalibFile, self.objectTracking, self.graphics, self.display, self.scale, self.pc_extrapolation)
    
    def __del__(self):
        self.sv.clean()


def main():
    kittiPath = sys.argv[1]
    scale_factor = int(sys.argv[2])
    pc_extrapolation = int(sys.argv[3])
    s = stereo_vision(width=1242//scale_factor, height=375//scale_factor, display=True, graphics=True, scale=scale_factor, pc_extrapolation=pc_extrapolation)
    
    for iFrame in range(465):
        leftName  = "{}/video/testing/image_02/0000/{:0>6}.png".format(kittiPath, iFrame)
        rightName = "{}/video/testing/image_03/0000/{:0>6}.png".format(kittiPath, iFrame)
        #print(leftName, rightName)
        left = cv2.imread(leftName)
        right = cv2.imread(rightName)
        left = cv2.resize(left, (left.shape[1]//scale_factor, left.shape[0]//scale_factor))
        right = cv2.resize(right, (right.shape[1]//scale_factor, right.shape[0]//scale_factor))
        s.generatePointCloud(left, right)

#if __name__ == "__main__":
#    main()
