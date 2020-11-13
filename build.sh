#!/bin/bash

if [ "$#" == "0" ]
	then
		echo "Expected 1 parameter : CUDA_MODE [ 0-CUDA_DISABLED / 1-CUDA_ENABLED ]"
		exit 1
fi

echo "CUDA_MODE : " $1

if [ $1 == "0" ]
	then
		echo "g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -pthread \`pkg-config --cflags --libs opencv\` -w"
		g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -pthread `pkg-config --cflags --libs opencv` -w
		echo "Compiled. Run the program using ./bin/stereo_vision -k path_to_kitti"	
		exit 0
fi


if [ $1 == "1" ]
	then
		echo "nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas/*.cpp  src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -Xcompiler="-pthread" \`pkg-config --cflags --libs opencv\` -w"
		nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas/*.cpp  src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -Xcompiler="-pthread" `pkg-config --cflags --libs opencv` -w
		echo "Compiled. Run the program using ./bin/stereo_vision -k path_to_kitti"
		exit 0
fi


echo "CUDA_MODE must be either 0 or 1"
exit 1
