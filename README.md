# Depth Perception from Stereoscopic Vision on Edge Devices

![demo](https://github.com/AdityaNG/Depth-Perception-from-Stereoscopic-Vision-on-Edge-Devices/blob/main/imgs/Point_Cloud_Outputs/Oct_29_2020/p2_final.gif?raw=true)


- Authors: [Dhruval PB](http://github.com/Dhruval360), [Aditya NG](http://github.com/AdityaNG)

## Dependencies

- [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
- A C++ compiler (*e.g.*, [GCC](http://gcc.gnu.org/))
- [LIBELAS](http://www.cvlibs.net/software/libelas/) 
- [OpenCV](https://github.com/opencv/opencv)

## Stereo Calibration

A calibrated pair of cameras is required for stereo rectification and calibration files should be stored in a `.yml` file. 
[This repository](https://github.com/sourishg/stereo-calibration) contains all the tools and instructions to calibrate stereo cameras.

The rotation and translation matrices for the point cloud transformation should be named as `XR` and `XT` in the calibration file. `XR` should be a **3 x 3** 
matrix and `XT` should be a **3 x 1** matrix. Please see a sample calibration file in the `calibration/` folder.

## Compiling

Clone the repository:

```bash
$ git clone https://github.com/...
```

Execute the `build.sh` script:

```bash
$ chmod +x build.sh
$ ./build.sh
```

## Running Dense 3D Reconstruction

```bash
$ chmod +x start.sh
$ ./start.sh
```

## License

This software is released under the [GNU GPL v3 license](LICENSE).