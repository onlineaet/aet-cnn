#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include "MtcsDeltaData.h"
#include "../DnnUtils.h"
#include "MtcsTool.h"

impl$ MtcsDeltaData{

   MtcsDeltaData(int size,int batch){
      self->size=size;
      self->batch = batch;
      dataArray = MtcsMem.calloc(batch*size*sizeof(float),TRUE);
      destroyFunc=MtcsMem.free;
      mean=NULL;
      variance=NULL;
   }

   public$ MtcsDeltaData(int w,int h,int channels,int batch){
      self->w=w;
      self->h=h;
      self->channels=channels;
      self->size=w*h*channels;
      self->batch = batch;
      dataArray = MtcsMem.calloc(batch*size*sizeof(float),TRUE);
      destroyFunc=MtcsMem.free;
      mean=MtcsMem.calloc(channels*sizeof(float),TRUE);
      variance=MtcsMem.calloc(channels*sizeof(float),TRUE);
   }

   void  setZero(){
      //MtcsMem.memset(dataArray,0,batch*size*sizeof(float));
      MtcsTool.fill(batch*size,0,dataArray);
   }

   //原型 scale_bias_kernel blas_kernels.cu
   __global__ void scale_bias_kernel(float *output, float *scale, int filters, int spatial, int current_size){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index >= current_size)
          return;
       int f = (index / spatial) % filters;
       output[index] *= scale[f];
   }

   //覆盖父类的方法scale
   void scale(float *scales){
      const int current_size = batch * channels * w*h;
      const int num_blocks = MtcsTool.getNumberOfBlocks(current_size, MTCS_BLOCK);
      scale_bias_kernel <<<num_blocks, MTCS_BLOCK, 0,MtcsTool.getStream() >>>(dataArray, scales, channels, w*h, current_size);
   }

   public$  void  scaleBias(float *scales){
      /* compare
      float *cpuScales=malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuScales,scales,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuDelta=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuDelta,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      */
      scale(scales);

      /* compare
      int i,j,b;
      int spatial=w*h;
      //#pragma omp parallel for  private( b,i,j)
      for(b = 0; b < batch; ++b){
         float *data=cpuDelta+b*size;
         for(i = 0; i < channels; ++i)
            for(j = 0; j < spatial; ++j)
               data[i*spatial + j] *= cpuScales[i];
      }
      float *compareDelta=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(compareDelta,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsDeltaData scaleBias 比较\n");
      DnnUtils.compare(batch*size,compareDelta,cpuDelta);
      free(cpuScales);
      free(cpuDelta);
      free(compareDelta);
      */
   }

   /**
    * cpu计算的是 sqrtf(variance[filter] + .00001f)
    * 原GPU是 .000001,有误差，现改为和cpu一样，误差在范围内
    */
   //原型 fast_mean_delta_kernel blas_kernels.cu
   __global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta){
      const int threads = MTCS_BLOCK;
      __shared__ float local[threads];

      int id = threadIdx.x;
      local[id] = 0;

      int filter = blockIdx.x;

      int i, j;
      for(j = 0; j < batch; ++j){
         for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
         }
      }
      __syncthreads();

      if(id == 0){
         mean_delta[filter] = 0;
         for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
         }
         mean_delta[filter] *= (-1.F/sqrtf(variance[filter] + .000001f));
      }
   }

   //原型 fast_mean_delta_gpu blas.h blas_kernels.cu
   void calcMean(float *variance){
      fast_mean_delta_kernel<<<channels, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,
            variance, batch, channels,w*h, mean);

      /* compare
      float *cpuMean=malloc(channels*sizeof(float));
      float *cpuDelta=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuDelta,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuVariance=malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuVariance,variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      //原型 mean_delta_cpu blas.h batchnorm_layer.c
      int filters=channels;
      int spatial=w*h;
      int i,j,k;
      //#pragma omp parallel for num_threads( needThreads ) private( i,j,k )
      for(i = 0; i < filters; ++i){
         cpuMean[i] = 0;
         for (j = 0; j < batch; ++j) {
            float *delta=cpuDelta+j*size+w*h*i;
            for (k = 0; k < spatial; ++k) {
               cpuMean[i] += delta[k];
            }
         }
         cpuMean[i] *= (-1./sqrt(cpuVariance[i] + .00001f));
      }

      float *compareMean=malloc(channels*sizeof(float));
      MtcsMem.memcpy(compareMean,mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsDeltaData calcMean 比较\n");
      DnnUtils.compare(channels,compareMean,cpuMean);
      free(cpuMean);
      free(cpuDelta);
      free(cpuVariance);
      free(compareMean);
      */
   }

   /**
    * cpu计算的是 powf(variance[filter] + .00001f
    * 原GPU是 .000001,有误差，现改为和cpu一样，误差在范围内
    */
   __global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance,
                              int batch, int filters, int spatial, float *variance_delta){
       const int threads = MTCS_BLOCK;
       __shared__ float local[threads];

       int id = threadIdx.x;
       local[id] = 0;

       int filter = blockIdx.x;

       int i, j;
       for(j = 0; j < batch; ++j){
           for(i = 0; i < spatial; i += threads){
               int index = j*spatial*filters + filter*spatial + i + id;

               local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
           }
       }
       __syncthreads();

       if(id == 0){
           variance_delta[filter] = 0;
           for(i = 0; i < threads; ++i){
               variance_delta[filter] += local[i];
           }
           variance_delta[filter] *= -.5 * powf(variance[filter] + .000001f, (float)(-3./2.));
       }
   }

   //原型  fast_variance_delta_gpu blas.h blas_kernels.cu
   void calcVariance(NormData *normData){
      fast_variance_delta_kernel<<<channels, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
            (normData->getX(), dataArray, normData->getMean(), normData->getVariance(), batch, channels,w*h,variance);

      /* compare
      float *cpuVariance = malloc(channels*sizeof(float));
      float *cpuX = malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuX, normData->getX(),batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuNormMean = malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuNormMean, normData->getMean(),channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuNormVariance = malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuNormVariance, normData->getVariance(),channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuDelta=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuDelta,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      int filters = channels;
      int spatial = w*h;
      int i,j,k;
      for(i = 0; i < filters; ++i){
         cpuVariance[i] = 0;
         for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + i*spatial + k;
               cpuVariance[i] += cpuDelta[index]*(cpuX[index] - cpuNormMean[i]);
            }
         }
         cpuVariance[i] *= -.5 * pow(cpuNormVariance[i] + .00001f, (float)(-3./2.));
      }
      float *compareCPU = malloc(channels*sizeof(float));
      MtcsMem.memcpy(compareCPU, variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsDeltaData calcVariance 比较\n");
      DnnUtils.compare(channels,compareCPU,cpuVariance);
      free(cpuVariance);
      free(compareCPU);
      free(cpuNormVariance);
      free(cpuNormMean);
      free(cpuDelta);
      free(cpuX);
      */
   }

   //原型 normalize_delta_gpu blas.h blas_kernels.cu
   __global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance,
         float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta){
       int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (index >= N) return;
       int f = (index/spatial)%filters;
       delta[index] = delta[index] * 1.F/(sqrtf(variance[f]) + .000001f)
             + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
   }

   //原型 normalize_delta_gpu blas.h blas_kernels.cu
   void normalize(NormData *normData){
      /* compare
      float *cpuDelta=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuDelta,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      */

      size_t totalSize = batch*channels*w*h;
      int spatial = w * h;
      normalize_delta_kernel<<<MtcsTool.gridSize(totalSize), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
      (totalSize, normData->getX(), normData->getMean(),
            normData->getVariance(), mean, variance,batch,channels,spatial, dataArray);

      /* compare
      float *cpuX = malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(cpuX, normData->getX(),batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuNormMean = malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuNormMean, normData->getMean(),channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuNormVariance = malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuNormVariance, normData->getVariance(),channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuMean = malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuMean, mean,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuVariance = malloc(channels*sizeof(float));
      MtcsMem.memcpy(cpuVariance, variance,channels*sizeof(float),MtcsCpyKind.DEV2HOST);
      int filters = channels;
      int f, j, k;
      for(j = 0; j < batch; ++j){
         for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + f*spatial + k;
               cpuDelta[index] = cpuDelta[index] * 1./(sqrt(cpuNormVariance[f] + .00001f))
                     + cpuVariance[f] * 2. * (cpuX[index] - cpuNormMean[f]) / (spatial * batch) + cpuMean[f]/(spatial*batch);
            }
         }
      }
      float *compareCpu=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(compareCpu,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsDeltaData normalize 比较\n");
      DnnUtils.compare(batch*size,compareCpu,cpuDelta);
      free(compareCpu);
      free(cpuVariance);
      free(cpuMean);
      free(cpuNormVariance);
      free(cpuNormMean);
      free(cpuX);
      free(cpuDelta);
      */

   }

   ~MtcsDeltaData(){
      if(mean){
         MtcsMem.free(mean);
         mean=NULL;
      }
       if(variance){
          MtcsMem.free(variance);
          variance=NULL;
       }
   }
};

