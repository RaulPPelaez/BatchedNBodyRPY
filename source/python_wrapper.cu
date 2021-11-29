/* Raul P. Pelaez 2020-2021. Python wrapper for the batched RPY Nbody evaluator.
   Three algorithms are provided:
     Fast: Leverages shared memory to hide bandwidth latency
     Naive: A dumb thread-per-partice parallelization of the N^2 double loop
     Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.
 */

#include"NbodyRPY.cuh"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
namespace py = pybind11;
using LayoutType = real3;
enum class algorithm{fast, naive, block};

void computeMdot_impl(py::array_t<real> h_pos, py::array_t<real> h_forces,
		      py::array_t<real> h_MF, int Nbatches, int NperBatch,
		      real selfMobility, real hydrodynamicRadius, algorithm alg) {
  constexpr size_t elementsPerValue = sizeof(LayoutType)/sizeof(real);
  const int numberParticles = Nbatches * NperBatch;
  cached_vector<real> pos(h_pos.data(), h_pos.data() + elementsPerValue*numberParticles);
  cached_vector<real> forces(h_forces.data(), h_forces.data() + elementsPerValue*numberParticles);
  cached_vector<real> Mv(elementsPerValue * numberParticles);
  auto kernel = computeRPYBatched<LayoutType>;
  if(alg==algorithm::naive)
    kernel = computeRPYBatchedNaive<LayoutType>;
  else if(alg==algorithm::block)
    kernel = computeRPYBatchedNaiveBlock<LayoutType>;
  
  kernel((LayoutType *)thrust::raw_pointer_cast(pos.data()),
	 (LayoutType *)thrust::raw_pointer_cast(forces.data()),
	 (LayoutType *)thrust::raw_pointer_cast(Mv.data()),
	 Nbatches, NperBatch, selfMobility, hydrodynamicRadius);
  thrust::copy(Mv.begin(), Mv.end(), h_MF.mutable_data());
}

void computeMdot(py::array_t<real> h_pos, py::array_t<real> h_forces,
                 py::array_t<real> h_MF, int Nbatches, int NperBatch,
                 real selfMobility, real hydrodynamicRadius) {
  computeMdot_impl(h_pos, h_forces, h_MF, Nbatches, NperBatch, selfMobility, hydrodynamicRadius, algorithm::fast);
}
void computeMdotNaive(py::array_t<real> h_pos, py::array_t<real> h_forces,
                 py::array_t<real> h_MF, int Nbatches, int NperBatch,
                 real selfMobility, real hydrodynamicRadius) {
  computeMdot_impl(h_pos, h_forces, h_MF, Nbatches, NperBatch, selfMobility, hydrodynamicRadius, algorithm::naive);
}
void computeMdotBlock(py::array_t<real> h_pos, py::array_t<real> h_forces,
                 py::array_t<real> h_MF, int Nbatches, int NperBatch,
                 real selfMobility, real hydrodynamicRadius) {
  computeMdot_impl(h_pos, h_forces, h_MF, Nbatches, NperBatch, selfMobility, hydrodynamicRadius, algorithm::block);
}


auto getPrecision(){
#ifndef DOUBLE_PRECISION
  constexpr auto precision = "float";
#else
  constexpr auto precision = "double";
#endif
  return precision;
}
using namespace pybind11::literals;
#ifndef MODULENAME
#define MODULENAME BatchedNBodyRPY
#endif
PYBIND11_MODULE(MODULENAME, m) {
  m.doc() = "NBody Batched RPY evaluator\nUSAGE:\n\tcall computeMdot, the input/output must have interleaved coordinates and each batch is placed after the previous one. [x_1_1 y_1_1 z_1_1,...x_1_NperBatch,...x_Nbatches_NperBatch]";  
  m.def("computeMdot", &computeMdot,
	"pos"_a, "force"_a, "MF"_a, "Nbatches"_a, "NperBatch"_a, "selfMobility"_a, "hydrodynamicRadius"_a);
  m.def("computeMdotNaive", &computeMdotNaive,
	"pos"_a, "force"_a, "MF"_a, "Nbatches"_a, "NperBatch"_a, "selfMobility"_a, "hydrodynamicRadius"_a);
  m.def("computeMdotBlock", &computeMdotBlock,
	"pos"_a, "force"_a, "MF"_a, "Nbatches"_a, "NperBatch"_a, "selfMobility"_a, "hydrodynamicRadius"_a);
  m.def("getPrecision", &getPrecision);
}
