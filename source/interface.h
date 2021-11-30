#ifndef NBODYRPY_INTERFACE_H
#define NBODYRPY_INTERFACE_H

#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
namespace uammd{
#if defined SINGLE_PRECISION
using  real  = float;
#else
using  real  = double;
#endif
}
enum class algorithm{fast, naive, block, advise};
void callBatchedNBodyRPY(const uammd::real* h_pos, const uammd::real* h_forces,
			 uammd::real* h_MF, int Nbatches, int NperBatch,
			 uammd::real selfMobility, uammd::real hydrodynamicRadius, algorithm alg);

#endif
