# Depth Perception from Stereoscopic Vision on Edge Devices

<img src="imgs/fsds.gif" height=400>

A library to simplify disparity calculation and 3D depth map generation from a stereo pair
- Authors: [Dhruval PB](http://github.com/Dhruval360), [Aditya NG](http://github.com/AdityaNG)

## About the Project

Depth estimation and 3D object detection are important for autonomous systems to be able to estimate their own state and gain greater context of their external environment. The project is an implementation of the software side of a perception stack.

<img src="imgs/single_loop.gif" height=400>

# Getting Started 

To quickly get the project running on your machine you can : 
- install the python library  OR
- compile from source

## Python Library [ coming soon ]

Install with pip
```bash
python3 -m pip install TODO
```

Run the sample program
```bash
python3 -m TODO
```

## Compiling

Clone the repository:

```bash
$ git clone https://github.com/AdityaNG/Depth-Perception-from-Stereoscopic-Vision-on-Edge-Devices.git
```

Compile using the make utility:

```bash
$ make stereo_vision -j$(($(nproc) * 2)) -s               # binary
$ make shared_library -j$(($(nproc) * 2)) -s              # shared object file
```

# TODO 

Things that we are currently working on

 - Rename the project  
 - README Update
 - Code clean up
 - Documentation
 - python wrapper
 - python library on pip
 - calibration functionality
 - Start using "issues" on github
 - Add CONTRIBUTING.md

# Stereo Calibration

A calibrated pair of cameras is required for stereo rectification and calibration files should be stored in a `.yml` file. 
[github.com/sourishg/stereo-calibration](https://github.com/sourishg/stereo-calibration) contains all the tools and instructions to calibrate stereo cameras.

The above should produce the camera intrinsic matrices `K1` and `K2` along with the distortion matrices `D1` and `D2`.
The extrinsic parameters of the stereo pair is calculated during runtime.

The rotation and translation matrices for the point cloud transformation should be named as `XR` and `XT` in the calibration file. `XR` should be a **3 x 3** 
matrix and `XT` should be a **3 x 1** matrix. Please see a sample calibration file in the `calibration/` folder.


# Dependencies

- [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
- A C++ compiler (*e.g.*, [G++](http://gcc.gnu.org/))
- [LIBELAS](http://www.cvlibs.net/software/libelas/) 
- [OpenCV](https://github.com/opencv/opencv)
- [Kitti Dataset](https://meet.google.com/linkredirect?authuser=0&dest=http%3A%2F%2Fwww.cvlibs.net%2Fdatasets%2Fkitti%2F)
- popt.h
- OpenGL

## License

This software is released under the [GNU GPL v3 license](LICENSE).
