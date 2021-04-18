# Depth Perception from Stereoscopic Vision on Edge Devices

![demo](https://github.com/AdityaNG/Depth-Perception-from-Stereoscopic-Vision-on-Edge-Devices/blob/main/imgs/single_loop.gif?raw=true)


- Authors: [Dhruval PB](http://github.com/Dhruval360), [Aditya NG](http://github.com/AdityaNG)

## Dependencies

- [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
- A C++ compiler (*e.g.*, [GCC](http://gcc.gnu.org/))
- [LIBELAS](http://www.cvlibs.net/software/libelas/) 
- [OpenCV](https://github.com/opencv/opencv)
- [Kitti Dataset](https://meet.google.com/linkredirect?authuser=0&dest=http%3A%2F%2Fwww.cvlibs.net%2Fdatasets%2Fkitti%2F)

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

Or compile using the make utility:

```bash
$ make stereo_vision -j$(nproc) -s
```
## License

This software is released under the [GNU GPL v3 license](LICENSE).
