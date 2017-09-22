# Grab the targets and sources as two batches
SOURCES = $(wildcard src/*.cxx)
HEADERS = $(wildcard include/*)
OBJECTS = $(patsubst src%.cxx,obj%.o,$(SOURCES))

# Versioning info
MAJOR=0
MINOR=0.0
SONAME=libbarcode.so
LIBNAME=$(SONAME).$(MAJOR).$(MINOR)
PREFIX=$(GM2FIELDSIGNALLIB)

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

FLAGS += -Wall -fPIC $(DEBUG) $(OPTIMIZE) -Iinclude -I$(GM2FIELDSIGNALLIB)/include

ifeq ($(USEGPUBARCODE),1)
ifdef CUDA_HOME
	LIBS += -L$(GM2FIELDSIGNALLIB)/lib -lm -lcudsp 
else
	LIBS += -L$(GM2FIELDSIGNALLIB)/lib -lm -ldsp 
endif
else
	LIBS += -L$(GM2FIELDSIGNALLIB)/lib -lm -ldsp 
endif

ifdef BOOST_INC
	FLAGS += -I$(BOOST_INC)
	LIBS += -L$(BOOST_LIB)
endif

LIBS += -lboost_filesystem -lboost_system

FLAGS += $(shell root-config --cflags)
LIBS  += $(shell root-config --libs)

all: lib/$(LIBNAME)

obj/%.o: src/%.cxx
	$(CXX) $(FLAGS) -o $@ -c $<

lib/$(LIBNAME): $(OBJECTS)
	$(CXX) -shared -fPIC $+ -o $@ $(LIBS)

install:
	mkdir -p $(PREFIX)/lib
	cp lib/$(LIBNAME) $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -r $(HEADERS) $(PREFIX)/include
	ln -sf $(PREFIX)/lib/$(LIBNAME) $(PREFIX)/lib/$(SONAME)
	$(LDCONFIG)

uninstall:
	rm -f $(PREFIX)/lib/$(SONAME)*
	rm -rf $(patsubst include/%,$(PREFIX)/include/%,$(HEADERS))

clean:
	rm -f $(TARGETS) obj/* lib/*
