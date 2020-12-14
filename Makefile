
PYBIND_ROOT=pybind11/
PYTHON=python3
NVCC=nvcc
#Uncomment for double precision, UAMMD is compiled in single by default
#DOUBLEPRECISION=-DDOUBLE_PRECISION
#In case you prefer to import with other name
MODULE_NAME=BatchedNBodyRPY
INCLUDE_FLAGS= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/

LIBRARY_NAME=$(MODULE_NAME)`$(PYTHON)-config --extension-suffix`
FILE=python_wrapper.cu
all:
	$(NVCC) -w -shared -std=c++11 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) -Xcompiler "-fPIC -w"  $(FILE) -o $(LIBRARY_NAME)
clean:
	rm $(LIBRARY_NAME)
