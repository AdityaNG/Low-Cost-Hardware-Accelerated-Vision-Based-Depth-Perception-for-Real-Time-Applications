#!/bin/bash

make clean
make stereo_vision -j10
make stereo_vision omp=1 -j10
make stereo_vision serial=1 -j10

echo Run each version with \"-v 1 -p 0\"