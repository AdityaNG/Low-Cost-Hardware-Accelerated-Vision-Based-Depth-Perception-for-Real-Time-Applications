UNAME := $(shell uname -m)




COMPILER := nvcc
SRC := src
OBJ := obj
SHARED_OBJ := shared
BIN := bin
EXECUTABLE := ${BIN}/stereo_vision
SHARED_LIBRARY := ${BIN}/stereo_vision.so
ELAS_DIR := ${SRC}/elas_cuda_openmp
FLAGS := -O3 -std=c++17 -w 
DEBUGFLAGS := -g -std=c++17 
LIBS := -lpopt -lglut -lGLU -lGL -lm `pkg-config --cflags --libs opencv` 

ifeq ($(UNAME), x86_64)
#	Default
endif
ifeq ($(UNAME), armv7l)
# 	Raspberry Pi
	LIBS := -lstdc++fs -lpopt -lglut -lGLU -lGL -lm `pkg-config --cflags --libs opencv4` 
endif

#OBJS := ${OBJ}/bayesian.o ${OBJ}/detector.o 
#OBJS := ${OBJ}/bayesian.o

$(shell mkdir -p ${BIN} ${OBJ} ${SHARED_OBJ})

ifeq ($(serial), 1)
	COMPILER := g++-8
	ELAS_DIR := ${SRC}/elas_openmp
	ELAS := $(wildcard $(ELAS_DIR)/*.cpp)
	ELAS_OBJS := 
	OBJS := ${OBJ}/stereo_vision.o ${ELAS_OBJS} ${OBJ}/graphing.o ${OBJS}
	SHARED_OBJS = $(patsubst $(OBJ)/%.o, $(SHARED_OBJ)/%.o, $(OBJS))
	LIBS := ${LIBS} -lpthread -fopenmp
	SHARED_FLAGS := ${FLAGS} -shared -fPIC -pie
else
	ifeq ($(old), 1)
		FLAGS := -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets ${FLAGS}
		DEBUGFLAGS := -gencode arch=compute_50,code=sm_50 ${DEBUGFLAGS}
	endif
	ELAS := $(wildcard $(ELAS_DIR)/*.cpp)
	ELAS_OBJS := $(patsubst $(ELAS_DIR)/%.cpp, $(OBJ)/%.o, $(ELAS)) $(OBJ)/elas_gpu.o ${OBJS}  
	OBJS := ${OBJ}/stereo_vision_v1.2.o ${ELAS_OBJS} ${OBJ}/graphing_gpu.o
	SHARED_OBJS = $(patsubst $(OBJ)/%.o, $(SHARED_OBJ)/%.o, $(OBJS))
	LIBS := ${LIBS} -Xcompiler="-pthread -fopenmp"
	SHARED_FLAGS := ${FLAGS} -shared --compiler-options="-fPIC -pie"
endif


stereo_vision: ${OBJS}
	${COMPILER} ${FLAGS} -o ${EXECUTABLE} ${OBJS} ${LIBS} && echo "Compiled Successfully!! Run the program using ./bin/stereo_vision -k path_to_kitti -v 1 -p 1 -f 1"

shared_library: FLAGS := ${SHARED_FLAGS}
shared_library: ${SHARED_OBJS}
	${COMPILER} ${FLAGS} -o ${SHARED_LIBRARY} $^ ${LIBS} && echo "Compiled the Shared Library Successfully!!"

debug: FLAGS := ${DEBUGFLAGS}
debug: ${OBJS}
	${COMPILER} ${FLAGS} -o ${EXECUTABLE} $^ ${LIBS} && echo "Compiled Successfully!! Run the program using ./bin/stereo_vision -k path_to_kitti -v 1 -p 1 -f 1"

%/bayesian.o: ${SRC}/bayesian/bayesian.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/descriptor.o: ${ELAS_DIR}/descriptor.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/elas.o: ${ELAS_DIR}/elas.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/filter.o: ${ELAS_DIR}/filter.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/matrix.o: ${ELAS_DIR}/matrix.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/triangle.o: ${ELAS_DIR}/triangle.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/elas_gpu.o: ${ELAS_DIR}/elas_gpu.cu
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/detector.o: ${SRC}/yolo/detector.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/graphing_gpu.o: ${SRC}/graphing/graphing.cu
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/graphing.o: ${SRC}/graphing/graphing.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/stereo_vision_v1.2.o: ${SRC}/stereo_vision_v1.2.cu
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

%/stereo_vision.o: ${SRC}/stereo_vision.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@ ${LIBS}

clean:
	rm -rf ${OBJ} ${SHARED_OBJ} ${BIN}