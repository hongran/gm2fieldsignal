# Grab the targets and sources as two batches
#SOURCES = $(wildcard src/*.cxx)
#EXE = $(patsubst src%.cxx,bin%,$(SOURCES))
SOURCES=src/TestFpsFid.cxx)
EXE=bin/TestFpsFid

# Figure out the architecture
UNAME_S = $(shell uname -s)

# Clang compiler
ifeq ($(UNAME_S), Darwin)
	CXX = clang++
	CC  = clang
	FLAGS = -std=c++11
endif

# Gnu compiler
ifeq ($(UNAME_S), Linux)
	CXX = g++
	CC  = gcc
	FLAGS = -std=c++11
endif

# Some optional compiler flags
ifdef DEBUG
	CC += -g -pg
	CXX += -g -pg
else
	CC += -O3
	CXX += -O3
endif

ifdef BOOST_INC
  FLAGS += -I$(BOOST_INC)
  LIBS += -L$(BOOST_LIB)
endif

LIBS += -lboost_filesystem -lboost_system

FLAGS += -Wall $(DEBUG) $(OPTIMIZE) -I$(GM2FIELDSIGNALLIB)/include
ifdef CUDA_HOME
	FLAGS += -I$(CUDA_HOME)/include
endif
FLAGS += -D"USEGPUFID=${USEGPUFID}"

LIBS += -L$(GM2FIELDSIGNALLIB)/lib -lm -lfid

FLAGS += $(shell root-config --cflags)
LIBS  += $(shell root-config --libs)

all: $(EXE) 

bin/%: src/%.cxx
	$(CXX) $(FLAGS) $(LIBS) -o $@ $<

clean:
	rm -f $(EXE)
