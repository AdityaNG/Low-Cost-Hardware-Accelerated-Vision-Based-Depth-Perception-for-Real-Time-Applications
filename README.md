# Low Cost Hardware Accelerated Vision Based Depth Perception for Real Time Applications

<p align="center">
    <img src="outputs/fsds.gif">
</p>

A library to simplify disparity calculation and 3D depth map generation from a stereo pair
- Authors: [Dhruval PB](http://github.com/Dhruval360), [Aditya NG](http://github.com/AdityaNG)

## About the Project

Depth estimation and 3D object detection are important for autonomous systems to be able to estimate their own state and gain greater context of their external environment. The project is an implementation of the software side of a perception stack.

<p align="center">
    <img src="outputs/PointCloud.gif">
</p>

# Quick Start

Install with pip
```bash
python3 -m pip install git+https://github.com/AdityaNG/Low-Cost-Hardware-Accelerated-Vision-Based-Depth-Perception-for-Real-Time-Applications
```

Run the demo
```bash
python3 -m stereo_vision --demo
```

# Dependencies

- [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
- A C++ compiler (*e.g.*, [G++](http://gcc.gnu.org/) or [Clang](https://clang.llvm.org/))
- [LIBELAS](http://www.cvlibs.net/software/libelas/) 
- [OpenCV](https://github.com/opencv/opencv)
- [Kitti Dataset](https://meet.google.com/linkredirect?authuser=0&dest=http%3A%2F%2Fwww.cvlibs.net%2Fdatasets%2Fkitti%2F)
- popt.h (for command line input)
- OpenGL
- Python 3 interpreter with all the packages in `requirements.txt` installed

# Stereo Calibration

A calibrated pair of cameras is required for stereo rectification and calibration files should be stored in a `.yml` file. 
[github.com/sourishg/stereo-calibration](https://github.com/sourishg/stereo-calibration) contains all the tools and instructions to calibrate stereo cameras.

The above should produce the camera intrinsic matrices `K1` and `K2` along with the distortion matrices `D1` and `D2`.
The extrinsic parameters of the stereo pair is calculated during runtime.

The rotation and translation matrices for the point cloud transformation should be named as `XR` and `XT` in the calibration file. `XR` should be a **3 x 3** 
matrix and `XT` should be a **3 x 1** matrix. Please see a sample calibration file in the `data/calibration/` folder.

# Compiling and running

Install the dependencies:

```bash
$ sudo apt install libpopt-dev freeglut3-dev       # popt.h and OpenGL
$ python3 -m pip install -r requirements.txt
```

Clone the repository:

```bash
$ git clone https://github.com/AdityaNG/Low-Cost-Hardware-Accelerated-Vision-Based-Depth-Perception-for-Real-Time-Applications
```

Compile using the make utility:

```bash
$ make stereo_vision -j$(($(nproc) * 2)) -s        # binary
$ make shared_library -j$(($(nproc) * 2)) -s       # shared object file
```
# TODO 

Things that we are currently working on

 - Code clean up
 - Extensive Documentation
 - Calibration functionality
 - Add CONTRIBUTING.md
 - Add "Good Starter" Issues
 - Understand and change LICENSE if necessary

## License

This software is released under the [GNU GPL v3 license](LICENSE).
