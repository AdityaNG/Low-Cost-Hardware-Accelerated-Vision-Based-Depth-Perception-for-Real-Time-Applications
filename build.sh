#!/bin/bash
g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/bayesian/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lstdc++fs -DWITH_FFMPEG=ON -lpopt -lglut -lGLU -lGL -std=c++11 -pthread `pkg-config --cflags --libs opencv` -w && echo "Compiled Successfully!! Run the program using ./bin/stereo_vision -k ../disparity/000000.png -v 4 -p 1 -f 1"
