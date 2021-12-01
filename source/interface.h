#ifndef NBODYRPY_INTERFACE_H
#define NBODYRPY_INTERFACE_H

#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
namespace nbody_rpy{
#if defined SINGLE_PRECISION
  using  real  = float;
#else
  using  real  = double;
#endif

  enum class algorithm{fast, naive, block, advise};
  void callBatchedNBodyRPY(const real* h_pos, const real* h_forces,
			   real* h_MF, int Nbatches, int NperBatch,
			   real selfMobility, real hydrodynamicRadius, algorithm alg);
}
#endif
