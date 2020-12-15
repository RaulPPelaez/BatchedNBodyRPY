/* Raul P. Pelaez 2020. Python wrapper for the batched RPY Nbody evaluator.
   If USE_NAIVE is defined, a dumb thread-per-particle algorithm will be used to do the computation for testing porpoises.
 */

#include"NbodyRPY.cuh"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
using LayoutType = real3;
void computeMdot(py::array_t<real> h_pos, py::array_t<real> h_forces,
                 py::array_t<real> h_MF, int Nbatches, int NperBatch,
                 real selfMobility, real hydrodynamicRadius) {
  constexpr size_t elementsPerValue = sizeof(LayoutType)/sizeof(real);
  const int numberParticles = Nbatches * NperBatch;
  cached_vector<real> pos(h_pos.data(), h_pos.data() + elementsPerValue*numberParticles);
  cached_vector<real> forces(h_forces.data(), h_forces.data() + elementsPerValue*numberParticles);
  cached_vector<real> Mv(elementsPerValue * numberParticles);
#if not defined USE_NAIVE
  constexpr auto kernel = computeRPYBatched<LayoutType>;
#else
  constexpr auto kernel = computeRPYBatchedNaive<LayoutType>;
#endif
  kernel((LayoutType *)thrust::raw_pointer_cast(pos.data()),
	 (LayoutType *)thrust::raw_pointer_cast(forces.data()),
	 (LayoutType *)thrust::raw_pointer_cast(Mv.data()),
	 Nbatches, NperBatch, selfMobility, hydrodynamicRadius);
  thrust::copy(Mv.begin(), Mv.end(), h_MF.mutable_data());
}

using namespace pybind11::literals;

PYBIND11_MODULE(BatchedNBodyRPY, m) {
  m.doc() = "NBody Batched RPY evaluator\nUSAGE:\n\tcall computeMdot, the input/output must have interleaved coordinates and each batch is placed after the previous one. [x_1_1 y_1_1 z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]";  
  m.def("computeMdot", &computeMdot,
	"pos"_a, "force"_a, "MF"_a, "Nbatches"_a, "NperBatch"_a, "selfMobility"_a, "hydrodynamicRadius"_a);
}
