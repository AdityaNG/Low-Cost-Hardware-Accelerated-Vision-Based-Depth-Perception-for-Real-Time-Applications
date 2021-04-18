COMPILER := nvcc
OBJ := obj
SRC := src
EXECUTABLE := bin/stereo_vision
LIBS := -lpopt -lglut -lGLU -lGL -lm `pkg-config --cflags --libs opencv` -Xcompiler="-pthread -fopenmp"
OBJS := ${OBJ}/bayesian.o ${OBJ}/descriptor.o ${OBJ}/elas.o ${OBJ}/filter.o ${OBJ}/matrix.o ${OBJ}/triangle.o ${OBJ}/elas_gpu.o ${OBJ}/detector.o ${OBJ}/graphing.o ${OBJ}/stereo_vision_v1.2.o
NVCCFLAGS ?=-O3 -std=c++11 -w # Has effect only if this variable hasn't been set externally


stereo_vision: ${OBJS}
	${COMPILER} ${NVCCFLAGS} -o ${EXECUTABLE} $^ ${LIBS} && echo "Compiled Successfully!! Run the program using ./bin/stereo_vision -k path_to_kitti -v 1 -p 0 -f 1"


${OBJ}/bayesian.o: ${SRC}/bayesian/bayesian.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/descriptor.o: ${SRC}/elas_cuda_openmp/descriptor.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/elas.o: ${SRC}/elas_cuda_openmp/elas.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/filter.o: ${SRC}/elas_cuda_openmp/filter.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/matrix.o: ${SRC}/elas_cuda_openmp/matrix.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/triangle.o: ${SRC}/elas_cuda_openmp/triangle.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/elas_gpu.o: ${SRC}/elas_cuda_openmp/elas_gpu.cu
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/detector.o: ${SRC}/yolo/detector.cpp
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/graphing.o: ${SRC}/graphing/graphing.cu
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

${OBJ}/stereo_vision_v1.2.o: ${SRC}/stereo_vision_v1.2.cu
	${COMPILER} ${NVCCFLAGS} -c $^ -o $@ ${LIBS}

clean:
	rm ${OBJ}/*.o ${EXECUTABLE}