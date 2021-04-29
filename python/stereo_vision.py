import ctypes
import os
import sys
import cv2
from numpy.ctypeslib import ndpointer
#import pandas as pd

def call_shared(L, function):
    arr = (ctypes.c_char_p * len(L))()
    arr[:] = L
    function(len(L), arr)

#os.chdir(os.path.dirname(os.path.abspath(__file__))) # Might be useful some other time
args = ["-k", "../kitti", "-v", "1", "-p", "0", "-f", "1"]
args = [s.encode('utf-8') for s in args]

sv = ctypes.CDLL('bin/stereo_vision.so')
#call_shared(args, sv.main)

sv.generatePointCloud.restype = ndpointer(dtype=ctypes.c_double, shape=(1242*375,3))

def generatePointCloud(left, right):
    left = cv2.cvtColor(left, cv2.COLOR_BGR2BGRA)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2BGRA)
    length = left.shape[0]
    width = left.shape[1]
    left = left.tostring()
    right = right.tostring()
    return sv.generatePointCloud(left, right, width, length)

if __name__ == "__main__":
    kittiPath = sys.argv[1]
    for iFrame in range(465):
        leftName  = "{}/video/testing/image_02/0000/{:0>6}.png".format(kittiPath, iFrame)
        rightName = "{}/video/testing/image_03/0000/{:0>6}.png".format(kittiPath, iFrame)
        #print(leftName, rightName)
        left = cv2.imread(leftName)
        right = cv2.imread(rightName)
        generatePointCloud(left, right)
    