#Raul P. Pelaez 2020-2021. Given a set of positions and forces acting on each particle, this module computes the product between the RPY tensor and the forces in the GPU.
#This module works on batches of particles, all batches must have the same size. Note that a batch size of 1 is equivalent to every particle interacting to every other.
#Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing an NPerBatch^2 matrix-vector products for each batch separately.
#The data layout is 3 interleaved coordinates with each batch placed after the previous one: [x_1_1, y_1_1, z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]
#Three algorithms are provided, each one works better in different regimes (a.i. number of particles and batch size).
#     Fast: Leverages shared memory to hide bandwidth latency
#     Naive: A dumb thread-per-partice parallelization of the N^2 double loop
#     Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.
#In general, testing suggests that Block is better for low sizes (less than 10K particles), Naive works best for intermediate sizes (10K-50K) and after that Fast is the better choice.
#As a side note, the reduction performed by Block is more accurate than the others, so while the results of Naive and Fast will be numerically identical, some differences due to numerical errors are expected between the latter and the former.
#You can profile this code with: nsys profile -w true --stats=true --trace nvtx,cuda   -o report --force-overwrite true python example.py 2>&1 | grep computeRPY
import BatchedNBodyRPY as rpy
import numpy as np
numberBatchs=1;
numberBlobsPerBatch=10000;
selfMobility= 1.0;
hydrodynamicRadius = 1.0;
np.random.seed(1234)
precision = np.float64 if rpy.getPrecision()=="double" else np.float32;

positions = np.array(np.random.rand(3*numberBatchs*numberBlobsPerBatch),precision);
forces = np.array(np.random.rand(3*numberBatchs*numberBlobsPerBatch),precision);
print("Positions[:9]: ", positions[:9])
print("Forces[:9]:", forces[:9])
Ntest=10
for i in range(0,Ntest):
    MFNaive=np.zeros(3*numberBatchs*numberBlobsPerBatch, precision);
    rpy.computeMdotNaive(positions, forces, MFNaive,
	                 numberBatchs, numberBlobsPerBatch,
	                 selfMobility, hydrodynamicRadius)

for i in range(0,Ntest):
    MF=np.zeros(3*numberBatchs*numberBlobsPerBatch, precision);
    rpy.computeMdot(positions, forces, MF,
	            numberBatchs, numberBlobsPerBatch,
	            selfMobility, hydrodynamicRadius)

for i in range(0,Ntest):
    MFBlock=np.zeros(3*numberBatchs*numberBlobsPerBatch, precision);
    rpy.computeMdotBlock(positions, forces, MFBlock,
                         numberBatchs, numberBlobsPerBatch,
                         selfMobility, hydrodynamicRadius)


err=MFBlock-MFNaive
print("Error between block and naive: (max and std)", np.max(err), np.std(err))
err=MF-MFNaive
print("Error between fast and naive: (max and std)", np.max(err), np.std(err))
