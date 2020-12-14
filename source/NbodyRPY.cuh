/* Raul P. Pelaez 2020. Batched Nbody evaluation of RPY kernel,
   Given N batches of particles (all with the same size NperBatch) the kernel nbodyFiberRPYGPU evaluates the matrix product RPY(ri, rj)*F ((NxN)*(Nx3) size) for all particles inside each batch.
 */
#ifndef NBODY_RPY_CUH
#define NBODY_RPY_CUH
#include<thrust/device_vector.h>
#include<iostream>
#include"allocator.h"
using resource = uammd::device_memory_resource;
using device_temporary_memory_resource = uammd::pool_memory_resource_adaptor<resource>;
template<class T> using allocator_thrust = uammd::polymorphic_allocator<T, device_temporary_memory_resource, thrust::cuda::pointer<T>>;
template<class T>  using cached_vector = thrust::device_vector<T, allocator_thrust<T>>;


#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif
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

//RPY = (f*I + g* r\diadic r/r^2). rh is hydrodynamic radius. This function returns {f, g/r^2}
inline __host__  __device__  real2 RPY(real r, real rh){
  const real invrh = real(1.0)/rh;
  r *= invrh;
  real2 c12;
  if(r >= real(2.0)){
    const real invr  = real(1.0)/r;
    const real invr2 = invr*invr;
    c12.x = (real(0.75) + real(0.5)*invr2)*invr;
    c12.y = (real(0.75) - real(1.5)*invr2)*invr*invr2;
  }
  else{
    c12.x = real(1.0)-real(0.28125)*r;
    c12.y = (r>real(0.0))?(real(0.09375)/r):real(0);
  }
  return {c12.x*invrh, c12.y*invrh};
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
__global__ void nbodyFiberRPYGPU(const vecType* pos,
				 const vecType* forces,
				 real3* Mv,
				 real selfMobility,
				 real hydrodynamicRadius,
				 int Nfibers,
				 int NperFiber){
  const int tid = blockIdx.x*blockDim.x+threadIdx.x;
  const int N = Nfibers*NperFiber;
  const bool active = tid < N;
  const int id = tid;
  const int fiber_id = thrust::min(int(blockIdx.x*blockDim.x)/NperFiber, Nfibers-1);
  const int blobsPerTile = blockDim.x;
  const int firstId = fiber_id*NperFiber;
  const int lastId =((firstId+blockDim.x)/NperFiber + 1)*NperFiber;
  const int tileOfFirstParticle = firstId/blobsPerTile;
  const int tileOfLastParticle = thrust::min(lastId/blobsPerTile+1, blobsPerTile);
  extern __shared__ char shMem[];
  real3 *shPos = (real3*) (shMem);
  real3 *shForce = (real3*) (shMem+blockDim.x*sizeof(real3));
  const real3 pi= active?make_real3(pos[id]):real3();
  real3 MF = real3();
  for(int tile = tileOfFirstParticle; tile<tileOfLastParticle; tile++){
    //Load tile to shared memory
    int i_load = tile*blockDim.x + threadIdx.x;
    if(i_load<N){
      shPos[threadIdx.x] = make_real3(pos[i_load]);
      shForce[threadIdx.x] = make_real3(forces[i_load]);
    }
    __syncthreads();
    //Compute interaction with all particles in tile
#pragma unroll 8
    for(uint counter = 0; counter<blockDim.x; counter++){
      if(!active) break;
      int cur_j = tile*blockDim.x + counter;
      int fiber_j = cur_j/NperFiber;
      if(fiber_id == fiber_j and cur_j<N){
	real3 fj = shForce[counter];
	real3 pj = shPos[counter];
	MF += computeRPYDotProductPair(pi, pj, fj, hydrodynamicRadius);	
      }
    }
    __syncthreads();
  }
  if(active)
    Mv[id] = selfMobility*MF;
}

template<class vecType>
void computeIntraFiberRPY(vecType* pos, vecType* force, real3 *Mv,
			  int Nfibers, int NperFiber, real selfMobility, real hydrodynamicRadius){
  int N = Nfibers*NperFiber;
  int nearestWarpMultiple = ((NperFiber+16)/32)*32;
  int minBlockSize = std::min(std::max(nearestWarpMultiple, 32), 512);
  int Nthreads = minBlockSize<N?minBlockSize:N;
  int Nblocks  = (N+Nthreads-1)/Nthreads;
  nbodyFiberRPYGPU<<<Nblocks, Nthreads, 2*Nthreads*sizeof(real3)>>>(pos,
								    force,
								    Mv,
								    selfMobility,
								    hydrodynamicRadius,
								    Nfibers, NperFiber);
  cudaDeviceSynchronize();
}

template<class vecType>
//Naive N^2 algorithm (looks like x20 times slower than the fast kernel
__global__ void nbodyFiberRPYGPUNaive(const vecType* pos,
				      const vecType* forces,
				      real3* Mv,
				      real selfMobility,
				      real hydrodynamicRadius,
				      int Nfibers,
				      int NperFiber){
  const int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid>=Nfibers*NperFiber) return;
  real3 pi = make_real3(pos[tid]);
  real3 MF = real3();
  int fiber_id = tid/NperFiber;
  printf("%g %g %g\n", pi.x, pi.y, pi.z);
  for(int i= fiber_id*NperFiber; i<(fiber_id+1)*NperFiber; i++){
    if(i>=Nfibers*NperFiber) break;
    real3 pj = make_real3(pos[i]);
    real3 fj = make_real3(forces[i]);    
    MF += computeRPYDotProductPair(pi, pj, fj, hydrodynamicRadius);
  }
  Mv[tid] = selfMobility*MF;
}

template<class vecType>
void computeIntraFiberRPYNaive(vecType* pos, vecType* force, real3 *Mv,
			       int Nfibers, int NperFiber,real selfMobility, real hydrodynamicRadius){
  int N = Nfibers*NperFiber;
  int minBlockSize = 128;
  int Nthreads = minBlockSize<N?minBlockSize:N;
  int Nblocks  = N/Nthreads;
  nbodyFiberRPYGPUNaive<<<Nblocks, Nthreads>>>(pos,
					       force,
					       Mv,
					       selfMobility,
					       hydrodynamicRadius,
					       Nfibers, NperFiber);
  cudaDeviceSynchronize();
}

#endif


#ifdef TEST_MODE

int main(){
  int Nfibers = 300;
  int NperFiber = 1000;
  thrust::device_vector<real4> pos(Nfibers*NperFiber);
  thrust::device_vector<real4> force(Nfibers*NperFiber);
  thrust::fill(force.begin(), force.end(), make_real4(1.0));
  thrust::fill(pos.begin(), pos.end(), make_real4(0));
  for(int i = 0; i< Nfibers*NperFiber; i++){
    pos[i] = make_real4(i%NperFiber, 0, 0, 0);
  } 
  thrust::device_vector<real3> Mv(Nfibers*NperFiber);
  real selfMobility= 1.0;
  real hydrodynamicRadius = 1.0;
  computeIntraFiberRPY(thrust::raw_pointer_cast(pos.data()),
		       thrust::raw_pointer_cast(force.data()),
		       thrust::raw_pointer_cast(Mv.data()),
		       Nfibers, NperFiber, selfMobility, hydrodynamicRadius);
  auto Mv_true = Mv;
  thrust::fill(Mv_true.begin(), Mv_true.end(), real3());
  computeIntraFiberRPYNaive(thrust::raw_pointer_cast(pos.data()),
			    thrust::raw_pointer_cast(force.data()),
			    thrust::raw_pointer_cast(Mv_true.data()),
			    Nfibers, NperFiber,selfMobility, hydrodynamicRadius);
  bool error = false;
  for(int i = 0; i<Nfibers*NperFiber; i++){
    real3 res = Mv[i];
    real3 rest = Mv_true[i];
    real4 p = pos[i];
    if(res.x-rest.x > 1e-5){       
      std::cout<<"ERROR: id: "<<i<<" fiber_id:  "<<i/NperFiber<<" difference with truth: "<<(res.x-rest.x)<<std::endl;
      error =true;
    }
  }
  if(not error)
    std::cout<<"ALL IS GOOD"<<std::endl;

return 0;
}

#endif

