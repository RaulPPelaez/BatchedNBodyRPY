/* Raul P. Pelaez 2020.

 */

#include"NbodyRPY.cuh"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
void computeMdot(py::array_t<real> h_pos, py::array_t<real> h_forces,
                 py::array_t<real> h_MF, int Nbatches, int NperBatch,
                 real selfMobility, real hydrodynamicRadius) {
  cached_vector<real> pos(3 * Nbatches * NperBatch);
  cached_vector<real> forces(3 * Nbatches * NperBatch);
  cached_vector<real> Mv(3 * Nbatches * NperBatch);
  int numberParticles = Nbatches * NperBatch;
  thrust::copy(h_pos.data(), h_pos.data() + 3*numberParticles, pos.begin());
  thrust::copy(h_forces.data(), h_forces.data() + 3*numberParticles, forces.begin());
  computeIntraFiberRPY((real3 *)thrust::raw_pointer_cast(pos.data()),
		       (real3 *)thrust::raw_pointer_cast(forces.data()),
		       (real3 *)thrust::raw_pointer_cast(Mv.data()),
		       Nbatches, NperBatch, selfMobility, hydrodynamicRadius);
  thrust::copy(Mv.begin(), Mv.end(), h_MF.mutable_data());
  cudaDeviceSynchronize();
}

using namespace pybind11::literals;

PYBIND11_MODULE(BatchedNBodyRPY, m) {
  m.doc() = "NBody Batched RPY evaluator\nUSAGE:\n\tcall computeMdot, the input/output must have interleaved coordinates and each batch is placed after the previous one. [x_1_1 y_1_1 z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]";  
  m.def("computeMdot", &computeMdot,
	"pos"_a, "force"_a, "MF"_a, "Nbatches"_a, "NperBatch"_a, "selfMobility"_a, "hydrodynamicRadius"_a);
}
