import ctypes
import os
import sys

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
sv = ctypes.CDLL('bin/stereo_vision.so')
sv.main(9, "-k", "../kitti", "-v", "1", "-p", "0", "-f", "1")