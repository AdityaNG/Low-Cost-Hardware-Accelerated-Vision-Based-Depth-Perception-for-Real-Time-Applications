import ctypes
import os
import sys

def call_shared(L, function):
    arr = (ctypes.c_char_p * len(L))()
    arr[:] = L
    function(len(L), arr)

#os.chdir(os.path.dirname(os.path.abspath(__file__))) # Might be useful some other time
args = ["-k", "../kitti", "-v", "1", "-p", "0", "-f", "1"]
args = [s.encode('utf-8') for s in args]

sv = ctypes.CDLL('bin/stereo_vision.so')
call_shared(args, sv.main)