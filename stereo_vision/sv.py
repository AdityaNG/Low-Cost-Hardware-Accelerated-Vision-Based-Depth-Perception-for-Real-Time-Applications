import ctypes
import os
import cv2
from numpy.ctypeslib import ndpointer
import numpy as np
import glob
import requests
import tqdm
import zipfile
import errno

def ensure_directory_exists(path: str):
    try:
        # `exist_ok` option is only available in Python 3.2+
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def unzip_file(src_path, unzip_path):
    """unzips file located at src_path into destination_path"""
    print("unzipping file...")

    # construct full path (including file name) for unzipping
    ensure_directory_exists(unzip_path)

    # extract data
    with zipfile.ZipFile(src_path, "r") as z:
        z.extractall(unzip_path)

    return True

def download_file(url: str, dest_path: str, show_progress_bars: bool = True):
    file_size = 0
    req = requests.get(url, stream=True)
    req.raise_for_status()

    # Total size in bytes.
    total_size = int(req.headers.get('content-length', 0))

    if os.path.exists(dest_path):
        print("target file already exists")
        file_size = os.stat(dest_path).st_size  # File size in bytes
        if file_size < total_size:
            # Download incomplete
            print("resuming download")
            resume_header = {'Range': 'bytes=%d-' % file_size}
            req = requests.get(url, headers=resume_header, stream=True,
                               verify=False, allow_redirects=True)
        elif file_size == total_size:
            # Download complete
            print("download complete")
            return
        else:
            # Error, delete file and restart download
            print("deleting file and restarting")
            os.remove(dest_path)
            file_size = 0
    else:
        # File does not exist, starting download
        print("starting download")

    # write dataset to file and show progress bar
    pbar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=dest_path, disable=not show_progress_bars)
    # Update progress bar to reflect how much of the file is already downloaded
    pbar.update(file_size)
    with open(dest_path, "ab") as dest_file:
        for chunk in req.iter_content(1024):
            dest_file.write(chunk)
            pbar.update(1024)

def normalize_depth(val, min_v, max_v):
    """ 
    print 'normalized depth value' 
    normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """
    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                         y < y_range[1], z > z_range[0], z < z_range[1]))]

def points_2_top_view(points, x_range, y_range, z_range, scale):
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2)
    
    # extract in-range points
    x_lim = in_range_points(x, x, y, z, x_range, y_range, z_range)
    y_lim = in_range_points(y, x, y, z, x_range, y_range, z_range)
    dist_lim = in_range_points(dist, x, y, z, x_range, y_range, z_range)
    
    # * x,y,z range are based on lidar coordinates
    x_size = int((y_range[1] - y_range[0]))
    y_size = int((x_range[1] - x_range[0]))
    
    # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
    # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    # scale - for high resolution
    x_img = -(y_lim * scale).astype(np.int32)
    y_img = -(x_lim * scale).astype(np.int32)

    # shift negative points to positive points (shift minimum value to 0)
    x_img += int(np.trunc(y_range[1] * scale))
    y_img += int(np.trunc(x_range[1] * scale))

    # normalize distance value & convert to depth map
    max_dist = np.sqrt((max(x_range)**2) + (max(y_range)**2))
    dist_lim = normalize_depth(dist_lim, min_v=0, max_v=max_dist)
    
    # array to img
    img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
    img[y_img, x_img] = dist_lim
    
    return img

site_packages_dir = "/".join(__file__.split("/")[:-2])
search_q = os.path.join(site_packages_dir,'stereo_vision*.so')
so_files = glob.glob(search_q)

DEFAULT_STEREO_VISION_SO_PATH = so_files[0]

KITTI_ZIP_PATH = os.path.join("/".join(__file__.split("/")[:-1]), 'data', 'kitti2015.zip')
KITTI_FOLDER_PATH = os.path.join("/".join(__file__.split("/")[:-1]), 'data', 'kitti2015')

class stereo_vision:

    def __init__(self, so_lib_path=DEFAULT_STEREO_VISION_SO_PATH, width=1242, height=375, 
    #def __init__(self, so_lib_path='bin/stereo_vision.so', width=1242, height=375, 
                defaultCalibFile=True, objectTracking=True, graphics=False, display=False, scale=1, pc_extrapolation=1,
                YOLO_CFG='src/yolo/yolov4-tiny.cfg', YOLO_WEIGHTS='src/yolo/yolov4-tiny.weights', YOLO_CLASSES='src/yolo/classes.txt',
                CAMERA_CALIBRATION_YAML='calibration/kitti_2011_09_26.yml'):
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
        
        self.YOLO_CFG = YOLO_CFG
        self.YOLO_WEIGHTS = YOLO_WEIGHTS
        self.YOLO_CLASSES = YOLO_CLASSES
        self.CAMERA_CALIBRATION_YAML = CAMERA_CALIBRATION_YAML
        self.sv.generatePointCloud.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        print(self.sv.generatePointCloud.argtypes)
        print(CAMERA_CALIBRATION_YAML)

    def generatePointCloud(self, left, right):
        left = cv2.cvtColor(left, cv2.COLOR_BGR2BGRA)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2BGRA)
        left = left.tostring()
        right = right.tostring()
        return self.sv.generatePointCloud(left, right, self.CAMERA_CALIBRATION_YAML.encode('utf-8'), self.width, self.height, self.defaultCalibFile, self.objectTracking, self.graphics, self.display, self.scale, self.pc_extrapolation,self.YOLO_CFG.encode('utf-8'), self.YOLO_WEIGHTS.encode('utf-8'), self.YOLO_CLASSES.encode('utf-8'))
    
    def __del__(self):
        self.sv.clean()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='stereo_vision CLI for disparity calculation and 3D depth map generation from a stereo pair')
    parser.add_argument('-k', '--kitti', type=str, default='~/KITTI', help='Path to KITTI directory of test images')
    parser.add_argument('-s', '--scale', type=int, default=1, help='By what factor to scale down the image by')
    parser.add_argument('-p', '--pointcloud_interpolation', default=False, action='store_true', help='TODO')
    
    parser.add_argument('-prl', '--parallel', default=False, action='store_true', help='Run parallel')

    parser.add_argument('-d', '--demo', default=False, action='store_true', help='Run the demo with the KITTI 2015 dataset')
    
    parser.add_argument('-c', '--camera_calibration', type=str, default=os.path.join("/".join(__file__.split("/")[:-1]) , 'data/kitti_2011_09_26.yml'), help='')
    parser.add_argument('-o', '--object_track', default=False, action='store_true', help='Enables Object Tracking with YOLO')
    parser.add_argument('-ycfg', '--yolo_cfg', type=str, default=os.path.join("/".join(__file__.split("/")[:-1]) , 'data/yolov4-tiny.cfg'), help='YOLO CFG file')
    parser.add_argument('-yw', '--yolo_weights', type=str, default=os.path.join("/".join(__file__.split("/")[:-1]) , 'data/yolov4-tiny.weights'), help='YOLO Weights file')
    parser.add_argument('-ycl', '--yolo_classes', type=str, default=os.path.join("/".join(__file__.split("/")[:-1]) , 'data/classes.txt'), help='YOLO Classes to track')
    
    parser.add_argument('-ctu', '--camera_to_use', default=-1, type=int, help='Enables camera use')
    parser.add_argument('-sw', '--swap', default=False, action='store_true', help='Swaps cameras')
    args = parser.parse_args()

    kittiPath = args.kitti # sys.argv[1]
    scale_factor = args.scale #int(sys.argv[2])
    pc_extrapolation = args.pointcloud_interpolation# int(sys.argv[3])
    
    so_file_path = DEFAULT_STEREO_VISION_SO_PATH
    if args.parallel:
        so_file_path = os.path.join("/".join(__file__.split("/")[:-1]) , 'bin/stereo_vision_parallel.so')

    CAMERA_CALIBRATION_YAML = os.path.join(os.getcwd(), args.camera_calibration)
    
    OBJ_TRACK = args.object_track
    YOLO_CFG = ""
    YOLO_WEIGHTS = ""
    YOLO_CLASSES = ""
    if OBJ_TRACK:
        YOLO_CFG = os.path.join(os.getcwd(), args.yolo_cfg)
        YOLO_WEIGHTS = os.path.join(os.getcwd(), args.yolo_weights)
        YOLO_CLASSES = os.path.join(os.getcwd(), args.yolo_classes)

    if args.demo:
        print("Downlading KITTI 2015 to ", KITTI_ZIP_PATH)
        download_file('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip', KITTI_ZIP_PATH)
        unzip_file(KITTI_ZIP_PATH, KITTI_FOLDER_PATH)
        
        s = stereo_vision(width=1242//scale_factor, height=375//scale_factor, objectTracking=False, display=True, graphics=True, scale=scale_factor, pc_extrapolation=pc_extrapolation, CAMERA_CALIBRATION_YAML = CAMERA_CALIBRATION_YAML, so_lib_path=so_file_path)
        #s = stereo_vision(width=1242//scale_factor, height=375//scale_factor, objectTracking=True, display=True, graphics=True, scale=scale_factor, pc_extrapolation=pc_extrapolation, YOLO_CFG=YOLO_CFG, YOLO_WEIGHTS=YOLO_WEIGHTS, YOLO_CLASSES=YOLO_CLASSES, so_lib_path=so_file_path)

        image_files = sorted(os.listdir(os.path.join(KITTI_FOLDER_PATH, 'testing', 'image_2')))
        for i in image_files:
            leftName = os.path.join(KITTI_FOLDER_PATH, 'testing', 'image_2', i)
            rightName = os.path.join(KITTI_FOLDER_PATH, 'testing', 'image_3', i)
            #print(leftName, rightName)
            left = cv2.imread(leftName)
            right = cv2.imread(rightName)
            left = cv2.resize(left, (left.shape[1]//scale_factor, left.shape[0]//scale_factor))
            right = cv2.resize(right, (right.shape[1]//scale_factor, right.shape[0]//scale_factor))
            s.generatePointCloud(left, right)

    if args.camera_to_use == -1:
        if OBJ_TRACK:
            s = stereo_vision(width=1242//scale_factor, height=375//scale_factor, objectTracking=OBJ_TRACK, display=True, graphics=True, scale=scale_factor, pc_extrapolation=pc_extrapolation, YOLO_CFG=YOLO_CFG, YOLO_WEIGHTS=YOLO_WEIGHTS, YOLO_CLASSES=YOLO_CLASSES, so_lib_path=so_file_path)
        else:
            s = stereo_vision(width=1242//scale_factor, height=375//scale_factor, objectTracking=False, display=True, graphics=True, scale=scale_factor, pc_extrapolation=pc_extrapolation, CAMERA_CALIBRATION_YAML = CAMERA_CALIBRATION_YAML, so_lib_path=so_file_path)
    
    
        for iFrame in range(465):
            leftName  = "{}/video/testing/image_02/0000/{:0>6}.png".format(kittiPath, iFrame)
            rightName = "{}/video/testing/image_03/0000/{:0>6}.png".format(kittiPath, iFrame)
            #print(leftName, rightName)
            left = cv2.imread(leftName)
            right = cv2.imread(rightName)
            left = cv2.resize(left, (left.shape[1]//scale_factor, left.shape[0]//scale_factor))
            right = cv2.resize(right, (right.shape[1]//scale_factor, right.shape[0]//scale_factor))
            s.generatePointCloud(left, right)
    else:
        camL = cv2.VideoCapture()
        camR = cv2.VideoCapture()
        if not(
            (camL.open(
                    args.camera_to_use)) and
            (camR.open(args.camera_to_use +2))):
            print(
            "Cannot open pair of system cameras connected \
                starting at camera #:",
                args.camera_to_use)
            exit()
        camL.grab()
        camR.grab()
        _, left = camL.retrieve()
        _, right = camR.retrieve()

        h, w, d = left.shape

        s = stereo_vision(width=w//scale_factor, height=h//scale_factor, objectTracking=False, display=True, graphics=True, scale=scale_factor, pc_extrapolation=pc_extrapolation, CAMERA_CALIBRATION_YAML = CAMERA_CALIBRATION_YAML, so_lib_path=so_file_path)
        
        while True:
            camL.grab()
            camR.grab()

            # then retrieve the images in slow(er) time
            # (do not be tempted to use read() !)

            _, left = camL.retrieve()
            _, right = camR.retrieve()

            if args.swap:
                tmp = left
                left = right
                right = tmp
            
            s.generatePointCloud(left, right)


#if __name__ == "__main__":
#    main()
