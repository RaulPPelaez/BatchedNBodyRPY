
PYBIND_ROOT=pybind11/
#This Makefile assumes the command $(PYTHON)-config exists
PYTHON=python3
NVCC=nvcc
#Uncomment for double precision, UAMMD is compiled in single by default
#DOUBLEPRECISION=-DDOUBLE_PRECISION
#In case you prefer to import with other name
MODULE_NAME=BatchedNBodyRPY
INCLUDE_FLAGS= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/

#Uncomment to use an alternative dumb algorithm to perform the computation (for testing correctness)
#NAIVE_ALGORITHM=-DUSE_NAIVE
LIBRARY_NAME=$(MODULE_NAME)`$(PYTHON)-config --extension-suffix`
FILE=python_wrapper.cu
all:
	$(NVCC) -O3 -w -shared -std=c++14 $(DOUBLEPRECISION) $(NAIVE_ALGORITHM) $(INCLUDE_FLAGS) -DMODULENAME=$(MODULE_NAME) -Xcompiler "-fPIC -w -O3"  $(FILE) -o $(LIBRARY_NAME)
clean:
	rm $(LIBRARY_NAME)
