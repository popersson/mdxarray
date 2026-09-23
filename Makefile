# mdxarray -- header-only; this builds the tests and examples only.
#
# libstdc++ does not ship <mdspan> yet, so libc++ is required on Linux.
CXX      = clang++
CXXFLAGS = -std=c++23 -stdlib=libc++ -O2 -Wall -Wextra
LDLIBS   = -lblas -llapack          # only examples needs these

all: test examples

# The test binary depends on nothing but the header.
test: test.cpp md.h
	$(CXX) $(CXXFLAGS) test.cpp -o $@

# The examples use the optional LAPACK shorthand, so they link BLAS/LAPACK.
examples: examples.cpp md.h md_lapack.h
	$(CXX) $(CXXFLAGS) examples.cpp -o $@ $(LDLIBS)

check: test
	./test

run: all
	./test && echo && ./examples

clean:
	rm -f test examples

.PHONY: all check run clean
