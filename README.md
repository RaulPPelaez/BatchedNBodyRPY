## Batched RPY evaluator

Raul P. Pelaez 2020-2021. Given a set of positions and forces acting on each particle, this module computes the product between the RPY tensor and the forces in the GPU.  
This module works on batches of particles, all batches must have the same size. Note that a batch size of 1 is equivalent to every particle interacting to every other.  
Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing an NPerBatch^2 matrix-vector products for each batch separately.  
The data layout is 3 interleaved coordinates with each batch placed after the previous one: ```[x_1_1, y_1_1, z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]```  
Three algorithms are provided, each one works better in different regimes (a.i. number of particles and batch size).  
	*Fast: Leverages shared memory to hide bandwidth latency  
	*Naive: A dumb thread-per-partice parallelization of the N^2 double loop  
	*Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.  
In general, testing suggests that Block is better for low sizes (less than 10K particles), Naive works best for intermediate sizes (10K-50K) and after that Fast is the better choice.  
As a side note, the reduction performed by Block is more accurate than the others, so while the results of Naive and Fast will be numerically identical, some differences due to numerical errors are expected between the latter and the former.  

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

Once compiled a BatchedNBodyRPY.*.so file will be available at the root directory.  
This allows to import it into python using   
```python
import BatchedNBodyRPY as rpy
```
Use help(rpy) for additional information.  
See example.py for a usage example.  
