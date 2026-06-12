#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <aet/lang/AString.h>
#include <aet/lang/AThread.h>
#include <aet/util/ARandom.h>
#include <aet/mtcs/MtcsSystem.h>
#include <aet/mtcs/MtcsStream.h>
#include <aet/mtcs/MtcsMem.h>

#include "MtcsTool.h"

impl$ MtcsTool{

   public$ static MtcsStream * getStream() {
       int i = MtcsSystem.getDevice();
       if (!streamInit[i]) {
           MtcsStream *stream=MtcsStream.buildStream(i);
           streamsArray[i] = stream;
           streamInit[i] = 1;
       }
       return streamsArray[i];
   }


   static __global__ void axpyKernel(int size, float alpha, float *src,float *dest){
       int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if(i < size)
          dest[i] += alpha*src[i];
   }

   static void axpy(int size, float alpha, float * src, float * dest){
      axpyKernel<<<MtcsTool.gridSize(size)/*!cuda_gridsize(channels)*/, MTCS_BLOCK, 0, MtcsTool.getStream()>>>
            (size, alpha,src, dest);
   }

   static __global__ void scalKernel(int size, float alpha, float *data){
       int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if(i < size)
          data[i] *= alpha;
   }

   static void scal(int size, float alpha, float *data){
      scalKernel<<<MtcsTool.gridSize(size)/*!cuda_gridsize(channels)*/, MTCS_BLOCK, 0, MtcsTool.getStream()>>>
            (size, alpha,data);
   }

   static __global__ void constrain(int totalSize,int size, float ALPHA, float **X, int INCX){
       int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if(i < totalSize){
          int b=i/size;
          int pos=i%size;
          X[b][pos*INCX] = fminf(ALPHA, fmaxf(-ALPHA, X[b][pos*INCX]));
       }
   }

   static __global__ void simpleCopyKernel(int size, float *src, float *dst){
       int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index < size)
           dst[index] = src[index];
   }

   //原型  simple_copy_ongpu blas.h blas_kernels.cu
   static void simpleCopy (int size, float *src, float *dst){
       const int num_blocks = size /MTCS_BLOCK + 1;
       simpleCopyKernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(size, src, dst);
   }


   static  __global__ void fix_nan_and_inf_kernel(float *input, size_t size){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index < size) {
           float val = input[index];
           if (isnan(val) || isinf(val)) {
               input[index] = 1.0f / (fabs((float)index) + 1);  // pseudo random value
           }
       }
   }

   //原型 fix_nan_and_inf blas.h blas_kernels.cu
   static void fixNanAndInf(float *input, size_t size){
       const int num_blocks = MtcsTool.getNumberOfBlocks(size, MTCS_BLOCK);
       fix_nan_and_inf_kernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(input, size);
   }

   static __global__ void reset_nan_and_inf_kernel(float *input, size_t size){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index < size) {
           float val = input[index];
           if (isnan(val) || isinf(val)) {
               input[index] = 0;
           }
       }
   }

   //原型 reset_nan_and_inf blas.h blas_kernels.cu
   static void resetNanAndInf(float *input, size_t size){
      const int num_blocks = MtcsTool.getNumberOfBlocks(size, MTCS_BLOCK);
      reset_nan_and_inf_kernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(input, size);
   }

   static __global__ void mulKernel(int N, float *X, float *Y){
      int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if(i < N)
         Y[i] *= X[i];
   }

   //魇型 mul_ongpu blas.h blas_kernels.cu
   static void mul(int N, float * X,float * Y){
      mulKernel<<<MtcsTool.gridSize(N), MTCS_BLOCK, 0, MtcsTool.getStream()>>>(N, X, Y);
   }

   //原型 fill_kernel blas_kernels.cu
   static __global__ void fillKernel(int N, float ALPHA, float *X){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index >= N)
         return;
      X[index] = ALPHA;
   }

   //原型 fill_ongpu blas.h blas_kernels.cu
   static void fill(int N, float ALPHA, float * X){
      const int num_blocks = MtcsTool.getNumberOfBlocks(N, MTCS_BLOCK);
      fillKernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(N, ALPHA, X);
   }

   //原型 constrain_kernel  blas_kernels.cu
   static __global__ void constrainKernel(int N, float ALPHA, float *X){
      int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if(i < N)
         X[i] = fminf(ALPHA, fmaxf(-ALPHA, X[i]));
   }

   //原型 constrain_ongpu blas.h blas_kernels.cu
   static void constrain(int N, float ALPHA, float * X){
      constrainKernel<<<MtcsTool.gridSize(N), MTCS_BLOCK, 0,  MtcsTool.getStream() >>>(N, ALPHA, X);
   }

   static __inline__ __device__  float warpAllReduceSum(float val) {
       for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
           val += __shfl_xor_sync_fs(0xffffffff, val, mask);
       return val;
   }


};

