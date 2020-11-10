#!/bin/bash

if [ "$#" == "0" ]
	then
		echo "Expected 1 parameter : CUDA_MODE [ 0-CUDA_DISABLED / 1-CUDA_ENABLED ]"
		exit 1
fi

echo "CUDA_MODE : " $1

if [ $1 == "0" ]
	then
		echo "g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/nlohmann/*.hpp  \`pkg-config --cflags --libs opencv\`"
		g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/graphing/*.cpp src/nlohmann/*.hpp -lglut -lGLU -lGL -std=c++11 -pthread `pkg-config --cflags --libs opencv`		
		exit 0
fi


if [ $1 == "1" ]
	then
		echo "nvcc -o bin/stereo_vision src/stereo_vision.cu src/stereo_vision.cpp src/elas/*.cpp  \`pkg-config --cflags --libs opencv\` -w"
		nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas/*.cpp  `pkg-config --cflags --libs opencv` -w
		exit 0
fi


echo "CUDA_MODE must be either 0 or 1"
exit 1
