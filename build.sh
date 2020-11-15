#!/bin/bash

if [ "$#" == "0" ]
	then
		echo "Expected 1 parameter : MODE [ 0-CPU / 1-CUDA / 2-OpenMP / 3-CUDA+OpenMP ]"
		exit 1
fi

echo "MODE : " $1

if [ $1 == "0" ]
	then
		echo "g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -pthread \`pkg-config --cflags --libs opencv\` -w"
		g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas/*.cpp src/bayesian/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lstdc++fs -DWITH_FFMPEG=ON -lpopt -lglut -lGLU -lGL -std=c++11 -pthread `pkg-config --cflags --libs opencv` -w
		echo "Compiled. Run the program using ./bin/stereo_vision -k path_to_kitti"	
		exit 0
fi


if [ $1 == "1" ]
	then
		echo "nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas/*.cpp  src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -Xcompiler="-pthread" \`pkg-config --cflags --libs opencv\` -w"
		nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas/*.cpp  src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -Xcompiler="-pthread" `pkg-config --cflags --libs opencv` -w && echo "Compiled. Run the program using ./bin/stereo_vision -k path_to_kitti"
		exit 0
fi

if [ $1 == "2" ]
	then
		echo "g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas_openmp/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -pthread -lm -fopenmp \`pkg-config --cflags --libs opencv\` -w"
		g++ -o bin/stereo_vision src/stereo_vision.cpp src/elas_openmp/*.cpp src/yolo/*.cpp src/graphing/*.cpp -lstdc++fs  -lpopt -lglut -lGLU -lGL -std=c++11 -pthread -lm -fopenmp `pkg-config --cflags --libs opencv` -w && echo "Compiled. Run the program using ./bin/stereo_vision -k path_to_kitti"	
		exit 0
fi

if [ $1 == "3" ]
	then
		echo "nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas_openmp/*.cpp  src/yolo/*.cpp src/graphing/*.cpp -lpopt -lglut -lGLU -lGL -std=c++11 -Xcompiler="-pthread -fopenmp" -lm \`pkg-config --cflags --libs opencv\` -w"
		nvcc -o bin/stereo_vision src/stereo_vision.cu src/elas_openmp/*.cpp  src/yolo/*.cpp src/graphing/*.cu -lpopt -lglut -lGLU -lGL -std=c++11 -Xcompiler="-pthread -fopenmp" -lm `pkg-config --cflags --libs opencv` -w && echo "Compiled. Run the program using ./bin/stereo_vision -k path_to_kitti"
		exit 0
fi

echo "MODE must be either 0 for CPU, 1 for CUDA, 2 for OpenMP, 3 for CUDA + OpenMP"
exit 1
