/* Raul P. Pelaez 2020-2021. Batched Nbody evaluation of RPY kernels,
   Given N batches of particles (all with the same size NperBatch) the kernel nbodyBatchRPYGPU evaluates the matrix product RPY(ri, rj)*F ((NperBatchxNperBatch)*(3xNperBatch) size) for all particles inside each batch.
   If DOUBLE_PRECISION is defined the code will be compiled in double.
   Three algorithms are provided:
     Fast: Leverages shared memory to hide bandwidth latency
     Naive: A dumb thread-per-partice parallelization of the N^2 double loop
     Block: Assigns a block to each particle, the first thread then reduces the result of the whole block.

 */
#ifndef NBODY_RPY_CUH
#define NBODY_RPY_CUH
#include<thrust/device_vector.h>
#include<iostream>
#include"allocator.h"
#include"interface.h"
using resource = uammd::device_memory_resource;
using device_temporary_memory_resource = uammd::pool_memory_resource_adaptor<resource>;
template<class T> using allocator_thrust = uammd::polymorphic_allocator<T, device_temporary_memory_resource, thrust::cuda::pointer<T>>;
template<class T>  using cached_vector = thrust::device_vector<T, allocator_thrust<T>>;

namespace uammd{
#if defined SINGLE_PRECISION
  using  real  = float;
  using  real2 = float2;
  using  real3 = float3;
  using  real4 = float4;
#else
  using  real  = double;
  using  real2 = double2;
  using  real3 = double3;
  using  real4 = double4;
#endif
}

#include"vector.cuh"

using namespace uammd;

//RPY = (1/(6*pi*viscosity*rh))*(f*I + g* r\diadic r/r^2). rh is hydrodynamic radius. This function returns {f, g/r^2}
inline __host__  __device__  real2 RPY(real r, real rh){
  const real invrh = real(1.0)/rh;
  r *= invrh;
  if(r >= real(2.0)){
    const real invr  = real(1.0)/r;
    const real invr2 = invr*invr;
    const real f = (real(0.75) + real(0.5)*invr2)*invr;
    const real ginvr2 = (real(0.75) - real(1.5)*invr2)*invr*invr2*invrh*invrh;
    return {f, ginvr2};
  }
  else{
    const real f = real(1.0)-real(0.28125)*r;
    const real ginvr2 = (r>real(0.0))?(real(0.09375)/(r*rh*rh)):real(0);
    return {f, ginvr2};
  }
}

//Computes M(ri, rj)*vj
__device__ real3 computeRPYDotProductPair(real3 pi, real3 pj, real3 vj, real rh){
  const real3 rij = make_real3(pi)-make_real3(pj);
  const real r = sqrt(dot(rij, rij));
  const real2 c12 = RPY(r, rh);
  const real f = c12.x;
  const real gdivr2 = c12.y;
  const real gv = gdivr2*dot(rij, vj);
  const real3 Mv_t = f*vj + (r>real(0)?gv*rij:real3());
  return Mv_t;
}

//Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3
//This kernel loads batches of particles into shared memory to speed up the computation.
//Threads will tipically read one value from global memory but blockDim.x from shared memory.
template<class vecType>
__global__ void computeRPYBatchedGPU(const vecType* pos,
				     const vecType* forces,
				     real3* Mv,
				     real selfMobility,
				     real hydrodynamicRadius,
				     int Nbatches,
				     int NperBatch){
  const int tid = blockIdx.x*blockDim.x+threadIdx.x;
  const int N = Nbatches*NperBatch;
  const bool active = tid < N;
  const int id = tid;
  const int fiber_id = thrust::min(tid/NperBatch, Nbatches-1);
  const int blobsPerTile = blockDim.x;
  const int firstId = blockIdx.x*blobsPerTile;
  const int lastId =((firstId+blockDim.x)/NperBatch + 1)*NperBatch;
  const int fiberOfFirstId = firstId/NperBatch;
  const int tileOfFirstParticle = fiberOfFirstId*NperBatch/blobsPerTile;
  const int numberTiles = N/blobsPerTile;
  const int tileOfLastParticle = thrust::min(lastId/blobsPerTile, numberTiles);
  extern __shared__ char shMem[];
  vecType *shPos = (vecType*) (shMem);
  vecType *shForce = (vecType*) (shMem+blockDim.x*sizeof(vecType));
  const real3 pi= active?make_real3(pos[id]):real3();
  real3 MF = real3();
  for(int tile = tileOfFirstParticle; tile<=tileOfLastParticle; tile++){
    //Load tile to shared memory
    int i_load = tile*blockDim.x + threadIdx.x;
    if(i_load<N){
      shPos[threadIdx.x] = make_real3(pos[i_load]);
      shForce[threadIdx.x] = make_real3(forces[i_load]);
    }
    __syncthreads();
    //Compute interaction with all particles in tile
    if(active){
#pragma unroll 8
      for(uint counter = 0; counter<blockDim.x; counter++){
	const int cur_j = tile*blockDim.x + counter;
	const int fiber_j = cur_j/NperBatch;
	if(fiber_id == fiber_j and cur_j<N){
	  const real3 fj = shForce[counter];
	  const real3 pj = shPos[counter];
	  MF += computeRPYDotProductPair(pi, pj, fj, hydrodynamicRadius);	
	}
      }
    }
    __syncthreads();
  }
  if(active)
    Mv[id] = selfMobility*MF;
}

template<class vecType>
void computeRPYBatched(vecType* pos, vecType* force, real3 *Mv,
			  int Nbatches, int NperBatch, real selfMobility, real hydrodynamicRadius){
  int N = Nbatches*NperBatch;
  int nearestWarpMultiple = ((NperBatch+16)/32)*32;
  int minBlockSize = std::max(nearestWarpMultiple, 32);
  int Nthreads = std::min(std::min(minBlockSize, N), 256);
  int Nblocks  = (N+Nthreads-1)/Nthreads;
  computeRPYBatchedGPU<<<Nblocks, Nthreads, 2*Nthreads*sizeof(real3)>>>(pos,
								    force,
								    Mv,
								    selfMobility,
								    hydrodynamicRadius,
								    Nbatches, NperBatch);
  cudaDeviceSynchronize();
}

template<class vecType>
//Naive N^2 algorithm (looks like x20 times slower than the fast kernel
__global__ void computeRPYBatchedNaiveGPU(const vecType* pos,
				      const vecType* forces,
				      real3* Mv,
				      real selfMobility,
				      real hydrodynamicRadius,
				      int Nbatches,
				      int NperBatch){
  const int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=Nbatches*NperBatch) return;
  real3 pi = make_real3(pos[tid]);
  real3 MF = real3();
  int fiber_id = tid/NperBatch;
  for(int i= fiber_id*NperBatch; i<(fiber_id+1)*NperBatch; i++){
    if(i>=Nbatches*NperBatch) break;
    real3 pj = make_real3(pos[i]);
    real3 fj = make_real3(forces[i]);
    MF += computeRPYDotProductPair(pi, pj, fj, hydrodynamicRadius);
  }
  Mv[tid] = selfMobility*MF;
}

template<class vecType>
void computeRPYBatchedNaive(vecType* pos, vecType* force, real3 *Mv,
			    int Nbatches, int NperBatch,real selfMobility, real hydrodynamicRadius){
  int N = Nbatches*NperBatch;
  int minBlockSize = 128;
  int Nthreads = minBlockSize<N?minBlockSize:N;
  int Nblocks  = N/Nthreads+1;
  computeRPYBatchedNaiveGPU<<<Nblocks, Nthreads>>>(pos,
						   force,
						   Mv,
						   selfMobility,
						   hydrodynamicRadius,
						   Nbatches, NperBatch);
  cudaDeviceSynchronize();
}


template<class vecType>
//NaiveBlock N^2 algorithm (looks like x20 times slower than the fast kernel
__global__ void computeRPYBatchedNaiveBlockGPU(const vecType* pos,
				      const vecType* forces,
				      real3* Mv,
				      real selfMobility,
				      real hydrodynamicRadius,
				      int Nbatches,
				      int NperBatch){
  const int tid = blockIdx.x;
  if(tid>=Nbatches*NperBatch) return;
  real3 pi = make_real3(pos[tid]);
  extern __shared__ real3 MFshared[];
  real3 MF = real3();
  int fiber_id = tid/NperBatch;
  int last_id = thrust::min((fiber_id+1)*NperBatch, Nbatches*NperBatch);
  for(int i= fiber_id*NperBatch+threadIdx.x; i<last_id; i+=blockDim.x){
      real3 pj = make_real3(pos[i]);
      real3 fj = make_real3(forces[i]);
      MF += computeRPYDotProductPair(pi, pj, fj, hydrodynamicRadius);
  }
  MFshared[threadIdx.x] = MF;
  __syncthreads();
  if(threadIdx.x == 0){
    auto MFTot = real3();
    for(int i =0; i<blockDim.x; i++){
      MFTot += MFshared[i];
    }
    Mv[tid] = selfMobility*MFTot;
  }

}

template<class vecType>
void computeRPYBatchedNaiveBlock(vecType* pos, vecType* force, real3 *Mv,
				 int Nbatches, int NperBatch,real selfMobility, real hydrodynamicRadius){
  int N = Nbatches*NperBatch;
  int minBlockSize = 128;
  int Nthreads = minBlockSize<N?minBlockSize:N;
  int Nblocks  = N;
  computeRPYBatchedNaiveBlockGPU<<<Nblocks, Nthreads, 2*Nthreads*sizeof(real3)>>>(pos,
										force,
										Mv,
										selfMobility,
										hydrodynamicRadius,
										Nbatches, NperBatch);
  cudaDeviceSynchronize();
}

namespace nbody_rpy{  
  using LayoutType = real3;
  void callBatchedNBodyRPY(const real* h_pos, const real* h_forces,
			   real* h_MF, int Nbatches, int NperBatch,
			   real selfMobility, real hydrodynamicRadius, algorithm alg){
    constexpr size_t elementsPerValue = sizeof(LayoutType)/sizeof(real);
    const int numberParticles = Nbatches * NperBatch;
    cached_vector<real> pos(h_pos, h_pos + elementsPerValue*numberParticles);
    cached_vector<real> forces(h_forces, h_forces + elementsPerValue*numberParticles);
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
    thrust::copy(Mv.begin(), Mv.end(), h_MF);
  }
}
#endif
