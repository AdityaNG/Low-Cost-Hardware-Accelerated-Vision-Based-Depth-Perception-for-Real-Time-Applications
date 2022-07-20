SRC            := src
BUILD          := build

OBJ            := ${BUILD}/obj
SHARED_OBJ     := ${BUILD}/shared
BIN            := ${BUILD}/bin

SRC_COMMON     := ${SRC}/common_includes
SRC_SERIAL     := ${SRC}/serial_includes
SRC_OMP        := ${SRC}/omp_includes
SRC_PARALLEL   := ${SRC}/parallel_includes

LIBS           := -lpopt -lglut -lGLU -lGL -lm `pkg-config --cflags --libs opencv`
FLAGS          := -O2 -std=c++17
DEBUGFLAGS     := -g -std=c++17  

SUBDIRECTORIES := $(sort $(dir $(wildcard $(SRC)/*/*/)))
DIRECTORIES    := $(patsubst $(SRC)/%, $(SHARED_OBJ)/%, $(SUBDIRECTORIES)) $(patsubst $(SRC)/%, $(OBJ)/%, $(SUBDIRECTORIES)) ${BIN}

$(shell mkdir -p ${DIRECTORIES})

ifneq (,$(findstring clean, $(MAKECMDGOALS))) # Prevent searching for compilers if the make target clean
else
	ifeq ($(video), 1)
		FLAGS := ${FLAGS} -DSHOW_VIDEO
	endif
	ifeq ($(profile), 1)
		FLAGS := ${FLAGS} -DPROFILE
	endif
	ifeq ($(serial), 1)
		CHECK := $(shell g++ --version >/dev/null 2>&1 || (echo "Failed to search for g++ with error: $$"))
		SRCS_COMMON := $(wildcard $(SRC_COMMON)/*/*.cpp) 
		SRCS_SERIAL := $(wildcard $(SRC_SERIAL)/*/*.cpp)

		OBJS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.cpp.o, $(SRCS_COMMON))  $(patsubst $(SRC)/%.cpp, $(OBJ)/%.cpp.o, $(SRCS_SERIAL))  
		SHARED_OBJS := $(patsubst $(SRC)/%.cpp, $(SHARED_OBJ)/%.cpp.o, $(SRCS_COMMON))  $(patsubst $(SRC)/%.cpp, $(SHARED_OBJ)/%.cpp.o, $(SRCS_SERIAL)) 
		FLAGS := ${FLAGS} -ffast-math

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
		SHARED_LIBRARY := ${BIN}/stereo_vision_serial.so
		
		LIBS := ${LIBS} -lpthread
		SHARED_FLAGS := ${FLAGS} -shared -fPIC
	
	else ifeq ($(omp), 1)
		CHECK := $(shell g++ --version >/dev/null 2>&1 || (echo "Failed to search for g++ with error: $$"))
		SRCS_COMMON := $(wildcard $(SRC_COMMON)/*/*.cpp) 
		SRCS_OMP := $(wildcard $(SRC_OMP)/*/*.cpp)

		OBJS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.cpp.o, $(SRCS_COMMON))  $(patsubst $(SRC)/%.cpp, $(OBJ)/%.cpp.o, $(SRCS_OMP))  
		SHARED_OBJS := $(patsubst $(SRC)/%.cpp, $(SHARED_OBJ)/%.cpp.o, $(SRCS_COMMON))  $(patsubst $(SRC)/%.cpp, $(SHARED_OBJ)/%.cpp.o, $(SRCS_OMP)) 
		FLAGS := ${FLAGS} -ffast-math

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
		EXECUTABLE := ${BIN}/stereo_vision_omp
		SHARED_LIBRARY := ${BIN}/stereo_vision_omp.so
		
		FLAGS := ${FLAGS} -fopenmp
		LIBS := ${LIBS} -lpthread
		SHARED_FLAGS := ${FLAGS} -shared -fPIC

	else
		SRCS_COMMON := $(wildcard $(SRC_COMMON)/*/*.cpp) 
		SRCS_PARALLEL := $(wildcard $(SRC_PARALLEL)/*/*.cpp) 
		SRCS_PARALLEL_CU := $(wildcard $(SRC_PARALLEL)/*/*.cu)

		OBJS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.cpp.o, $(SRCS_COMMON))  $(patsubst $(SRC)/%.cpp, $(OBJ)/%.cpp.o, $(SRCS_PARALLEL)) $(patsubst $(SRC)/%.cu, $(OBJ)/%.cu.o, $(SRCS_PARALLEL_CU)) 
		SHARED_OBJS := $(patsubst $(SRC)/%.cpp, $(SHARED_OBJ)/%.cpp.o, $(SRCS_COMMON))  $(patsubst $(SRC)/%.cpp, $(SHARED_OBJ)/%.cpp.o, $(SRCS_PARALLEL)) $(patsubst $(SRC)/%.cu, $(SHARED_OBJ)/%.cu.o, $(SRCS_PARALLEL_CU)) 

		CHECK := $(shell nvcc --version >/dev/null 2>&1 || (echo "Failed to search for nvcc with error: $$?"))
			ifeq (,${CHECK})
$(info CUDA/C++ compiler found: nvcc)
$(info )
			COMPILER := nvcc
			else
$(error No CUDA/C++ compilers found. Either install cuda-toolkit or try compiling the serial version)
			endif
		EXECUTABLE := ${BIN}/stereo_vision_parallel
		SHARED_LIBRARY := ${BIN}/stereo_vision_parallel.so
		
		ifeq ($(old), 1)
			FLAGS := -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets ${FLAGS}
			DEBUGFLAGS := -gencode arch=compute_50,code=sm_50 ${DEBUGFLAGS}
		endif
		
		SHARED_FLAGS := ${FLAGS} -shared --compiler-options="-fPIC -pie -ffast-math" -Xcompiler="-pthread -fopenmp"
		FLAGS := ${FLAGS} --compiler-options="-ffast-math" -Xcompiler="-pthread -fopenmp"
	endif
endif

stereo_vision: ${OBJS}
	@echo
	${COMPILER} ${FLAGS} -o ${EXECUTABLE} ${OBJS} ${LIBS} 
	@echo
	@echo "Compiled Successfully!! Run the program using ./${EXECUTABLE} -k path_to_kitti -v 1"

shared_library: FLAGS := ${SHARED_FLAGS}
shared_library: ${SHARED_OBJS}
	@echo
	${COMPILER} ${FLAGS} -o ${SHARED_LIBRARY} $^ ${LIBS} 
	@echo
	@echo "Compiled the Shared Library Successfully!! You can find it at: ${SHARED_LIBRARY}"

debug: FLAGS := ${DEBUGFLAGS}
debug: ${OBJS}
	@echo
	${COMPILER} ${FLAGS} -o ${EXECUTABLE} $^ ${LIBS} 
	@echo
	@echo "Compiled Successfully!! Run the program using ./${EXECUTABLE} -k path_to_kitti -v 1"

${OBJ}/%.cu.o: ${SRC}/%.cu
	${COMPILER} ${FLAGS} -c $^ -o $@

${OBJ}/%.cpp.o: ${SRC}/%.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

${SHARED_OBJ}/%.cu.o: ${SRC}/%.cu
	${COMPILER} ${FLAGS} -c $^ -o $@

${SHARED_OBJ}/%.cpp.o: ${SRC}/%.cpp
	${COMPILER} ${FLAGS} -c $^ -o $@

clean:
	rm -rf ${BUILD}