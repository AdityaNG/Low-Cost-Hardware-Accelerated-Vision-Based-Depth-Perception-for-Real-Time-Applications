###
# Plotter Code
###
#
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph import Transform3D
import numpy as np
import sys
import threading

global THREADS
THREADS = dict()

global POINTS
POINTS = []

global graph_region
graph_region = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), color=(0, 1, 0, 0.5), size=0.05, pxMode=False)

global BOXES, MAX_BOXES
BOXES = []
MAX_BOXES = 10
for i in range(MAX_BOXES):
    box = gl.GLBoxItem(size=QtGui.QVector3D(3,2,1), color=(255,0,0,100), glOptions="opaque")
    BOXES.append(box)

def reset_BOXES():
    global BOXES, MAX_BOXES
    for i in range(MAX_BOXES):
        BOXES[i].translate(0,0,0)


reset_BOXES()
#box2 = gl.GLBoxItem(size=QtGui.QVector3D(2,2,5), color=(0,255,0,100), glOptions="opaque")
#box2.translate(30,8,3)

#box = gl.GLVolumeItem(np.array([1,1,1,1], dtype=np.ubyte), glOptions="opaque")
# ['GLAxisItem', 'GLBarGraphItem', 'GLBoxItem', 'GLGraphicsItem', 'GLGridItem', 
# 'GLImageItem', 'GLLinePlotItem', 'GLMeshItem', 'GLScatterPlotItem', 
# 'GLSurfacePlotItem', 'GLViewWidget', 'GLVolumeItem', 'MeshData', '__builtins__',
# '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'items', 'np', 'shaders']
print(dir(gl))


def update_graph():
    global graph_region, POINTS, COLORS, BOXES
    #colors = np.array(shape=(len(POINTS), 4), dtype=np.float32)
    #print(COLORS)
    colors = np.array(COLORS, dtype=np.float32)
    if len(POINTS) > 0:
        a = np.array(POINTS)

        #a[:, 0] = a[:, 0] * 0.9
        #a[:, 1] = a[:, 1] * 0.9
        #a[:, 2] = a[:, 2] * 1.5
        #a[:, 1] = a[:, 1] * -1
        graph_region.setData(pos=a, color=colors)
        # graph_region.setData(pos=np.flip(np.array(POINTS), 2), color=colors)


def start_graph():
    print("Setting up graph")
    global app, graph_region, w, g, d3, t
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.resize(800, 600)
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('LIDAR Point Cloud')

    g = gl.GLGridItem()
    #g.translate(0,0,-1)
    w.addItem(g)


    # pos3 = np.zeros((100, 100, 3))
    # pos3[:, :, :2] = np.mgrid[:100, :100].transpose(1, 2, 0) * [-0.1, 0.1]
    # pos3 = pos3.reshape(10000, 3)
    # d3 = (pos3 ** 2).sum(axis=1) ** 0.5

    #graph_region = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), color=(0, 1, 0, 0.5), size=0.001, pxMode=False)
    # graph_region.rotate(180, 1, 0, 0)
    # graph_region.translate(0, 0, 2.4)
    w.addItem(graph_region)

    for i in range(MAX_BOXES):
        w.addItem(BOXES[i])
    
    t = QtCore.QTimer()
    t.timeout.connect(update_graph)
    t.start(50)


    QtGui.QApplication.instance().exec_()
    global RUNNING
    RUNNING = False
    print("\n[STOP]\tGraph Window closed. Stopping...")


import os
import time
import random
from math import sin, cos, pi, atan

global last_reload
last_reload = 0 #time.time()
def reload_check():
    global last_reload
    statbuf = os.stat('plotter/3D_maps/reload_check')
    if statbuf.st_mtime>=last_reload:
        print("Reloading",statbuf.st_mtime)
        last_reload = time.time()#statbuf.st_mtime
        return True
    return False


def validPoint(x,y,z):
    return True;
    return z>=0;


def start_scan():
    global POINTS, COLORS, BOXES
    try:
        while True:
            #if input("Reload [y/n]")=="y":
            if reload_check():
                reset_BOXES()
                POINTS = []
                COLORS = []
                file1 = open('plotter/3D_maps/pnts.csv', 'r') 
                Lines = file1.readlines() 
                POINTS_start = False
                POINTS_end = False
                BOXES_start = False
                BOXES_end = False
                BOX_COUNT = 0
                #for line in Lines: 
                for i in range(len(Lines)): 
                    line = Lines[i]
                    
                    if not POINTS_start:
                        if "POINTS" in line:
                            POINTS_start = True

                    if not POINTS_end:
                        if "POINTS_END" in line:
                            POINTS_end = True

                    if not BOXES_start:
                        if "BOXES" in line:
                            BOXES_start = True

                    if not BOXES_end:
                        if "BOXES_END" in line:
                            BOXES_end = True
                    
                    if POINTS_start and not POINTS_end:
                        if ("]" in line):
                            x, y, z = [float(Lines[i-2].replace("[","").replace(";","").replace("\n","")),float(Lines[i-1].replace(";","").replace("\n","")),float(Lines[i].split("]")[0].replace("]","").replace("\n",""))]
                            r, g, b = list(map(lambda c: float(c)/255.0, Lines[i].split("]")[1].split(" ")))
                            
                            if validPoint(x,y,z) and (r<0.9 and g<0.9 and b<0.9):
                                COLORS.append((r, g, b, 1))
                                POINTS.append([x, y, z])

                    if BOXES_start and not BOXES_end:
                        if ("]" in line):
                            x, y, z = [float(Lines[i-2].replace("[","").replace(";","").replace("\n","")),float(Lines[i-1].replace(";","").replace("\n","")),float(Lines[i].split("]")[0].replace("]","").replace("\n",""))]
                            r, g, b = list(map(lambda c: float(c)/255.0, Lines[i].split("]")[1].split(" ")))
                            
                            if validPoint(x,y,z) and BOX_COUNT<MAX_BOXES:
                                #BOXES.append([x, y, z])
                                BOXES[BOX_COUNT].translate(x,y,z, local=False)
                                BOX_COUNT += 1
                            #print(x,y,z)
                print("Done")
            time.sleep(1)
    except Exception as e:
        print(e)
        

serial_thread = threading.Thread(target=start_scan, args=())
serial_thread.daemon = True  # Daemonize thread
THREADS["SERIAL"] = serial_thread

for t in THREADS:
    THREADS[t].start()


if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        graph_thread = threading.Thread(target=start_graph, args=())
        graph_thread.daemon = True  # Daemonize thread
        # thread.start()  # Start the execution this is done after we have connected to the simulator
        #THREADS["GRAPH"] = graph_thread
        start_graph()
