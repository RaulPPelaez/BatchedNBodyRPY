#Given a set of positions and forces acting on each one, this module computes the product between the RPY tensor and the forces.
#This module works on batches of particles, all batches must have the same size.
#Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing an NPerBatch^2 matrix-vector products for each batch separately.
#The data layout is 3 interleaved coordinates with each batch placed after the previous one: [x_1_1, y_1_1, z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]
import BatchedNBodyRPY as rpy
import numpy as np
numberBatchs=2;
numberBlobsPerBatch=3;
selfMobility= 1.0;
hydrodynamicRadius = 1.0;

precision = np.float32;
#Just copies of the same fiber laid in the x axis
fiber_xpos = np.array(range(1, numberBlobsPerBatch+1), precision);
fiber_ypos = fiber_xpos*0;
fiber_zpos = fiber_xpos*0;
positions = np.tile(np.ravel((fiber_xpos, fiber_ypos, fiber_zpos), order='F'), numberBatchs);
#forces = np.array(np.ones(3*numberBatchs*numberBlobsPerBatch), precision);
#All particles have a force in the X direction
forces = np.array((positions!=0)*1, precision)
print("Positions[:9]: ", positions[:9])
print("Forces[:9]:", forces[:9])
#It is really important that the result array has the same floating precision as the compiled module, otherwise
# python will just silently pass by copy and the results will be lost
MF=np.zeros(3*numberBatchs*numberBlobsPerBatch, precision);

rpy.computeMdot(positions, forces, MF,
                numberBatchs, numberBlobsPerBatch,
                selfMobility, hydrodynamicRadius)
#MF now contains the product between the RPY tensor and the forces.
print("MF[:9] in the x direction")
print(MF[:9:3])
print("MF[:9] in the y direction")
print(MF[1:9:3])



