# Grab the targets and sources as two batches
SOURCES = $(wildcard src/*.cxx)
HEADERS = $(wildcard include/*)
OBJECTS = obj/dsp_cpu.o

GPU_SOURCES = $(wildcard src/*.cu)
GPU_OBJECTS = obj/dsp_gpu.o


# Versioning info
MAJOR=0
MINOR=0.0
PREFIX=$(GM2FIELDSIGNALLIB)
SONAME=libdsp.so
LIBNAME=$(SONAME).$(MAJOR).$(MINOR)
GPU_SONAME=libcudsp.so
GPU_LIBNAME=$(GPU_SONAME).$(MAJOR).$(MINOR)

# Figure out the architecture
UNAME_S = $(shell uname -s)

# Clang compiler
ifeq ($(UNAME_S), Darwin)
	CXX = clang++
	CC  = clang
	FLAGS = -std=c++11
	LDCONFIG = cp $(PREFIX)/lib/$(LIBNAME) $(PREFIX)/lib/$(SONAME).$(MAJOR)
endif

# Gnu compiler
ifeq ($(UNAME_S), Linux)
	CXX = g++
	CC  = gcc
	FLAGS = -std=c++11
	LDCONFIG = ldconfig -n -v $(PREFIX)/lib
endif

# Some optional compiler flags
ifdef DEBUG
	CC += -g -pg
	CXX += -g -pg
else
	CC += -O3
	CXX += -O3
endif

FLAGS += -Wall -fPIC $(DEBUG) -Iinclude

LIBS += -lm -lfftw3

ROOT_FLAGS += $(shell root-config --cflags)
ROOT_LIBS  += $(shell root-config --libs)

GPU_FLAGS := -std=c++11 -I$(CUDA_HOME)/include -Iinclude
GPU_FLAGS += --compiler-options '-Wall -fPIC'
GPU_FLAGS += -m64 
GPU_LIBS += -L$(CUDA_HOME)/lib64 -lcudart -lcufft -lcusolver -lcublas -lm -lgomp

NVCC := nvcc -ccbin g++

GENCODE := -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

ifdef CUDA_HOME
  all: lib/$(LIBNAME) lib/$(GPU_LIBNAME)
else
  all: lib/$(LIBNAME)
endif

obj/%.o: src/%.cxx $(HEADERS)
	$(CXX) $(FLAGS) $(ROOT_FLAGS) -o $@ -c $<

ifdef CUDA_HOME
obj/%.o: src/%.cu $(HEADERS)
	$(NVCC)  $(GPU_FLAGS) $(GENCODE) -o $@ -c $<

lib/$(GPU_LIBNAME): $(GPU_OBJECTS)
	$(CXX) -shared -fPIC $+ -o $@ $(GPU_LIBS)
endif

lib/$(LIBNAME): $(OBJECTS)
	$(CXX) -shared -fPIC $+ -o $@ $(LIBS)

ifdef CUDA_HOME
install:
	mkdir -p $(PREFIX)/lib
	cp lib/$(LIBNAME) $(PREFIX)/lib
	cp lib/$(GPU_LIBNAME) $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -r $(HEADERS) $(PREFIX)/include
	ln -sf $(PREFIX)/lib/$(LIBNAME) $(PREFIX)/lib/$(SONAME)
	ln -sf $(PREFIX)/lib/$(GPU_LIBNAME) $(PREFIX)/lib/$(GPU_SONAME)
	$(LDCONFIG)
else
install:
	mkdir -p $(PREFIX)/lib
	cp lib/$(LIBNAME) $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -r $(HEADERS) $(PREFIX)/include
	ln -sf $(PREFIX)/lib/$(LIBNAME) $(PREFIX)/lib/$(SONAME)
	$(LDCONFIG)
endif


uninstall:
	rm -f $(PREFIX)/lib/$(SONAME)*
	rm -rf $(patsubst include/%,$(PREFIX)/include/%,$(HEADERS))
clean:
	rm -f obj/* lib/*
