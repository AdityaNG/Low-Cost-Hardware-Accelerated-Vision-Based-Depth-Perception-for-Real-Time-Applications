#!/bin/bash

make clean
make stereo_vision profile=1 -j10
make stereo_vision profile=1 omp=1 -j10
make stereo_vision profile=1 serial=1 -j10

echo Run each version with \"-v 1 -p 0\"