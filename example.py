#Given a set of positions and forces acting on each, this module computes the product between the RPY tensor and the forces.
#This module works on batches of particles, all batches must have the same size.
#Only the elements of the mobility matrix that correspond to pairs that belong to the same batch are non zero. It is equivalent to computing an NPerBatch^2 matrix-vector products for each batch separately.
import BatchedNBodyRPY as rpy
import numpy as np
numberFibers=2;
numberBlobsPerFiber=3;
selfMobility= 1.0;
hydrodynamicRadius = 1.0;

#Just copies of the same fiber laid in the x axis
fiber_xpos = np.array(range(0, numberBlobsPerFiber), np.float32);
fiber_ypos = fiber_xpos*0;
fiber_zpos = fiber_xpos*0;
positions = np.tile(np.ravel((fiber_xpos, fiber_ypos, fiber_zpos), order='F'), numberFibers);
forces = np.array(np.ones(3*numberFibers*numberBlobsPerFiber), np.float32);

#It is really important that the result array has the same floating precision as the compiled uammd, otherwise
# python will just silently pass by copy and the results will be lost
MF=np.zeros(3*numberFibers*numberBlobsPerFiber, np.float32);

rpy.computeMdot(positions, forces, MF, numberFibers, numberBlobsPerFiber, selfMobility, hydrodynamicRadius)
#MF now contains the product between the RPY tensor and the forces.
print(MF)



