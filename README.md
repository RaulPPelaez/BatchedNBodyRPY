## Batched RPY evaluator

Given a set of positions and forces acting on each one, this module computes the product between the RPY tensor and the forces.  
This module works on batches of particles, all batches must have the same size.  
Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing a matrix-vector products for each batch separately.  
The data layout is 3 interleaved coordinates with each batch placed after the previous one: ```[x_1_1, y_1_1, z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]```  

## REQUIREMENTS

In order to compile this repo you need [pybind11](https://github.com/pybind/pybind11), it is already included as a submodule, so simply clone this repo recursively to get it:  
```shell
git clone --recursive https://github.com/stochasticHydroTools/DPPoissonTests
```
You need the CUDA toolkit installed, I tested up to the latest version at the moment (CUDA 11).  
Finally a working python environment is required.  

## USAGE 

Compile using ```make```, you may have to customize sources/Makefile to adapt it with your system paths.  
There are a couple options in the Makefile:  
	-DOUBLEPRECISION: Compiles the module in double precision (single by default)  
	-NAIVE_ALGORITHM: Uses a dumb Nbody algorithm to evaluate the kernel (useful for testing correctness) but much much slower.  

Once compiled a BatchedNBodyRPY.*.so file will be available at the root directory.  
This allows to import it into python using   
```python
import BatchedNBodyRPY as rpy
```
Use help(rpy) for additional information.  
See example.py for a usage example.  
