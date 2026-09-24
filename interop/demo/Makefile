# Demo library for the md_abi interop layer.
CXX      = clang++
CXXFLAGS = -std=c++23 -stdlib=libc++ -O2 -Wall -Wextra -fPIC -I../..

all: libmesh.so

libmesh.so: libmesh.cpp ../../md_abi.h ../../md.h
	$(CXX) $(CXXFLAGS) -shared libmesh.cpp -o $@

check: libmesh.so
	julia demo.jl
	python3 demo.py

clean:
	rm -f libmesh.so

.PHONY: all check clean
