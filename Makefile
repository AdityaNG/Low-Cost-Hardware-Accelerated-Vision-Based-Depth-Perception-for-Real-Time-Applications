BUILD := build
ROOTSERIAL := ${BUILD}/serial
ROOTPARALLEL := ${BUILD}/parallel
SRC := src
OBJ := obj
SHARED_OBJ := shared
BIN := ${BUILD}/bin
SHARED_LIBRARY := ${BIN}/stereo_vision.so
ELAS_DIR := ${SRC}/elas_cuda_openmp
FLAGS := -O3 -std=c++17 -w
DEBUGFLAGS := -g -std=c++17 
LIBS := -lpopt -lglut -lGLU -lGL -lm `pkg-config --cflags --libs opencv`  

$(shell mkdir -p ${BUILD} ${BIN} ${ROOTPARALLEL}/${OBJ} ${ROOTPARALLEL}/${SHARED_OBJ} ${ROOTSERIAL}/${OBJ} ${ROOTSERIAL}/${SHARED_OBJ})

ifeq ($(serial), 1)
	CHECK := $(shell g++ --version >/dev/null 2>&1 || (echo "Failed to search for g++ with error: $$"))
	ifeq (,${CHECK})
    	COMPILER := g++
$(info C++ compiler found: g++)
$(info )

	else # Check for clang if g++ is unavailable
    	CHECK2 := $(shell clang --version >/dev/null 2>&1 || (echo "Failed to search for clang with error: $$?"))
		ifeq (,${CHECK2})
$(info C++ compiler found: clang)
$(info )
		COMPILER := clang
		LIBS := -lstdc++ ${LIBS}
		else
$(error No C++ compilers found.)
		endif
	endif
	EXECUTABLE := ${BIN}/stereo_vision_serial
	ROOT := ${ROOTSERIAL}
	OBJ := $(ROOT)/$(OBJ)
	SHARED_OBJ := $(ROOT)/$(SHARED_OBJ)
	OBJS := ${OBJ}/bayesian.o ${OBJ}/detector.o
	ELAS_DIR := ${SRC}/elas_openmp
	ELAS := $(wildcard $(ELAS_DIR)/*.cpp)
	ELAS_OBJS := $(patsubst $(ELAS_DIR)/%.cpp, $(OBJ)/%.o, $(ELAS))
	OBJS := ${OBJ}/stereo_vision.o ${ELAS_OBJS} ${OBJ}/graphing.o ${OBJS}
	SHARED_OBJS = $(patsubst $(OBJ)/%.o, $(SHARED_OBJ)/%.o, $(OBJS))
	LIBS := ${LIBS} -lpthread -fopenmp
	SHARED_FLAGS := ${FLAGS} -shared -fPIC
else
	CHECK := $(shell nvcc --version >/dev/null 2>&1 || (echo "Failed to search for nvcc with error: $$?"))
		ifeq (,${CHECK})
$(info CUDA/C++ compiler found: nvcc)
$(info )
		COMPILER := nvcc
		else
$(error No CUDA/C++ compilers found. Either install cuda-toolkit or try compiling the serial version)
		endif
	EXECUTABLE := ${BIN}/stereo_vision
	ROOT := ${ROOTPARALLEL}
	OBJ := $(ROOT)/$(OBJ)
	SHARED_OBJ := $(ROOT)/$(SHARED_OBJ)
	OBJS := ${OBJ}/bayesian.o ${OBJ}/detector.o
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
	${COMPILER} ${FLAGS} -o ${EXECUTABLE} ${OBJS} ${LIBS} 
	@echo
	@echo "Compiled Successfully!! Run the program using ./${EXECUTABLE} -k path_to_kitti -v 1 -p 1 -f 1"

shared_library: FLAGS := ${SHARED_FLAGS}
shared_library: ${SHARED_OBJS}
	${COMPILER} ${FLAGS} -o ${SHARED_LIBRARY} $^ ${LIBS} 
	@echo
	@echo "Compiled the Shared Library Successfully!!"

debug: FLAGS := ${DEBUGFLAGS}
debug: ${OBJS}
	${COMPILER} ${FLAGS} -o ${EXECUTABLE} $^ ${LIBS} 
	@echo
	@echo "Compiled Successfully!! Run the program using ./${EXECUTABLE} -k path_to_kitti -v 1 -p 1 -f 1"

%/bayesian.o: ${SRC}/bayesian/bayesian.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/descriptor.o: ${ELAS_DIR}/descriptor.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/elas.o: ${ELAS_DIR}/elas.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/filter.o: ${ELAS_DIR}/filter.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/matrix.o: ${ELAS_DIR}/matrix.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/triangle.o: ${ELAS_DIR}/triangle.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/elas_gpu.o: ${ELAS_DIR}/elas_gpu.cu
	${COMPILER} ${FLAGS} -c $^ -o $@

%/detector.o: ${SRC}/yolo/detector.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/graphing_gpu.o: ${SRC}/graphing/graphing.cu
	${COMPILER} ${FLAGS} -c $^ -o $@

%/graphing.o: ${SRC}/graphing/graphing.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

%/stereo_vision_v1.2.o: ${SRC}/stereo_vision_v1.2.cu
	${COMPILER} ${FLAGS} -c $^ -o $@

%/stereo_vision.o: ${SRC}/stereo_vision.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

clean:
	rm -rf ${BUILD}