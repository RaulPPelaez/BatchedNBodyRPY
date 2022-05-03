
PYBIND_ROOT=pybind11/
#This Makefile assumes the command $(PYTHON)-config exists
PYTHON=python3
CXX=g++
NVCC=nvcc -ccbin=$(CXX)
#Uncomment for double precision, UAMMD is compiled in single by default
#DOUBLEPRECISION=-DDOUBLE_PRECISION
#In case you prefer to import with other name
MODULE_NAME=BatchedNBodyRPY
INCLUDE_FLAGS:= `$(PYTHON)-config --includes` -I $(PYBIND_ROOT)/include/
LIBRARY_NAME:=../$(MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)

all: $(LIBRARY_NAME)

NbodyRPY.o: NbodyRPY.cu
	$(NVCC) -O3 -c -std=c++14 $(DOUBLEPRECISION) -Xcompiler "-fPIC -O3"  $^ -o $@

python_wrapper.o: python_wrapper.cpp NbodyRPY.o
	$(CXX) -fPIC -O3 -c -std=c++14  $(DOUBLEPRECISION) $(INCLUDE_FLAGS) -DMODULENAME=$(MODULE_NAME) $< -o $@

$(LIBRARY_NAME): NbodyRPY.o python_wrapper.o
	$(NVCC) -O3 -w -shared $^ -o $@
clean:
	rm -f $(LIBRARY_NAME) NbodyRPY.o python_wrapper.o
