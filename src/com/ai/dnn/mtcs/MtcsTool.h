#ifndef __COM_AI_CNN_MTCS_TOOL_H__
#define __COM_AI_CNN_MTCS_TOOL_H__

#include <aet.h>
#include <aet/mtcs/MtcsStream.h>
#include "../cnnmicro.h"

package$ com.ai.dnn.mtcs;


public$ class$ MtcsTool{
   static MtcsStream * streamsArray[16];    // cudaStreamSynchronize( get_cuda_stream() );
   static int streamInit[16];
   public$  static MtcsStream * getStream();
   public$  static __global__  void simpleCopy(int totalSize,int size,float **dest, float **src);
   private$ static __global__ void copyKernel(int totalSize,int size,float **dest, float **src);
   public$  static void copy(int totalSize,int size, float ** dest, float **src);
   private$ static  __global__ void axpyKernel(int size, float ALPHA, float *dest,float *src);
   //原型 axpy_ongpu blas.h blas_kernels.cu
   public$  static void axpy(int size, float alpha, float * src, float * dest);
   private$ static __global__ void scalKernel(int size, float alpha, float *data);
   public$  static void scal(int size, float alpha, float *data);
   private$ static __global__ void mulKernel(int N, float *X, float *Y);
   //魇型 mul_ongpu blas.h blas_kernels.cu
   public$ static void mul(int N, float * X,float * Y);
   public$  static __global__ void constrain(int totalSize,int size, float ALPHA, float **X, int INCX);
   //原型 simple_copy_kernel blas_kernels.cu
   private$ static __global__ void simpleCopyKernel(int size, float *src, float *dst);
   //原型  simple_copy_ongpu blas.h blas_kernels.cu
   public$ static void simpleCopy (int size, float *src, float *dst);
   //原型 fix_nan_and_inf blas.h blas_kernels.cu
   private$ static  __global__ void fix_nan_and_inf_kernel(float *input, size_t size);
   public$ static void fixNanAndInf(float *input, size_t size);
   public$ static __global__ void reset_nan_and_inf_kernel(float *input, size_t size);
   //原型 reset_nan_and_inf blas.h blas_kernels.cu
   public$ static void resetNanAndInf(float *input, size_t size);
   //原型 fill_kernel blas_kernels.cu
   public$ static __global__ void fillKernel(int N, float ALPHA, float *X);
   //原型 fill_ongpu blas.h blas_kernels.cu
   public$ static void fill(int N, float ALPHA, float * X);
   //原型 constrain_kernel  blas_kernels.cu
   private$ static __global__ void constrainKernel(int N, float ALPHA, float *X);
   //原型 constrain_ongpu blas.h blas_kernels.cu
   public$ static void constrain(int N, float ALPHA, float * X);

   public$ static dim3 gridSize(size_t n){
      size_t k = (n-1) / MTCS_BLOCK + 1;
      size_t x = k;
      size_t y = 1;
      if(x > 65535){
         x = ceil(sqrt(k));
         y = (n-1)/(x*MTCS_BLOCK) + 1;
      }
      //dim3 d = { (unsigned int)x, (unsigned int)y, 1 };
      dim3 d;
      d.x = x;
      d.y = y;
      d.z = 1;
      //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
      return d;
   }

   //原型 get_number_of_blocks dark_cuda.h dark_cuda.c
   public$ static int getNumberOfBlocks(int arraySize, int blockSize){
       return arraySize / blockSize + ((arraySize % blockSize > 0) ? 1 : 0);
   }

   public$  static __inline__ __device__  float warpAllReduceSum(float val);


};




#endif /* __COM_AI_CNN_MTCS_TOOL_H__ */

