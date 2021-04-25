COMPILER := nvcc
SRC := src
OBJ := obj
SHARED_OBJ := shared
BIN := bin
EXECUTABLE := ${BIN}/stereo_vision
SHARED_LIBRARY := ${BIN}/stereo_vision.so
LIBS := -lpopt -lglut -lGLU -lGL -lm `pkg-config --cflags --libs opencv` -Xcompiler="-pthread -fopenmp"
OBJS := ${OBJ}/bayesian.o ${OBJ}/descriptor.o ${OBJ}/elas.o ${OBJ}/filter.o ${OBJ}/matrix.o ${OBJ}/triangle.o ${OBJ}/elas_gpu.o ${OBJ}/detector.o ${OBJ}/graphing.o ${OBJ}/stereo_vision_v1.2.o
SHARED_OBJS = ${SHARED_OBJ}/bayesian.o ${SHARED_OBJ}/descriptor.o ${SHARED_OBJ}/elas.o ${SHARED_OBJ}/filter.o ${SHARED_OBJ}/matrix.o ${SHARED_OBJ}/triangle.o ${SHARED_OBJ}/elas_gpu.o ${SHARED_OBJ}/detector.o ${SHARED_OBJ}/graphing.o ${SHARED_OBJ}/stereo_vision_v1.2.o

$(shell mkdir -p ${BIN} ${OBJ} ${SHARED_OBJ})

ifeq ($(old), 1)
	NVCCFLAGS := -gencode arch=compute_50,code=sm_50 -O3 -std=c++11 -w -Wno-deprecated-gpu-targets
else
	NVCCFLAGS := -O3 -std=c++11 -w
endif

ifeq ($(shared), 1)
	NVCCFLAGS := ${NVCCFLAGS} -shared --compiler-options="-fPIC -pie"
endif

ifeq ($(debug), 1)
	NVCCFLAGS := ${NVCCFLAGS} -g
endif

stereo_vision: ${OBJS}
	${COMPILER} ${NVCCFLAGS} -o ${EXECUTABLE} $^ ${LIBS} && echo "Compiled Successfully!! Run the program using ./bin/stereo_vision -k path_to_kitti -v 1 -p 0 -f 1"

shared_library: ${SHARED_OBJS}
	${COMPILER} ${NVCCFLAGS} -o ${SHARED_LIBRARY} $^ ${LIBS} && echo "Compiled the Shared Library Successfully!!"

%/bayesian.o: ${SRC}/bayesian/bayesian.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/descriptor.o: ${SRC}/elas_cuda_openmp/descriptor.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/elas.o: ${SRC}/elas_cuda_openmp/elas.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/filter.o: ${SRC}/elas_cuda_openmp/filter.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/matrix.o: ${SRC}/elas_cuda_openmp/matrix.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/triangle.o: ${SRC}/elas_cuda_openmp/triangle.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/elas_gpu.o: ${SRC}/elas_cuda_openmp/elas_gpu.cu
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/detector.o: ${SRC}/yolo/detector.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/graphing.o: ${SRC}/graphing/graphing.cu
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

%/stereo_vision_v1.2.o: ${SRC}/stereo_vision_v1.2.cu
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

clean:
	rm -rf ${OBJ} ${SHARED_OBJ} ${BIN}