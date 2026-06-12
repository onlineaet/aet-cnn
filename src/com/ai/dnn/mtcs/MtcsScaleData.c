#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/time/Time.h>
#include <aet/lang/System.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include "MtcsScaleData.h"
#include "MtcsTool.h"
#include "../DnnUtils.h"


impl$ MtcsScaleData{

   MtcsScaleData(int size){
      self->size=size;
      scales=MtcsMem.calloc(size*sizeof(float),TRUE);
      scale_updates=MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   void   init(float value){
      float *temp=malloc(size*sizeof(float));
      int i;
      for(i=0;i<size;i++)
           temp[i]=value;
      MtcsMem.memcpy(scales,temp,size*sizeof(float),MtcsCpyKind.HOST2DEV);
      free(temp);
   }

   void freeData(){
      if(scales){
         MtcsMem.free(scales);
         scales=NULL;
      }
      if(scale_updates){
         MtcsMem.free(scale_updates);
         scale_updates=NULL;
      }
   }

   public$ void   createEma(){
      if(!ema)
         ema=MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   public$ void   createUpdates(){
      if(!scale_updates)
         scale_updates=MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   //魇型 backward_scale_kernel blas_kernels.cu
   __global__ void backward_scale_kernel(float *x_norm, float *deltas, int batch, int n, int size, float *scale_updates){
      __shared__ float part[MTCS_BLOCK];
      int i,b;
      int filter = blockIdx.x;
      int p = threadIdx.x;
      float sum = 0;
      for(b = 0; b < batch; ++b){
         for(i = 0; i < size; i += MTCS_BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? deltas[index]*x_norm[index] : 0;
         }
      }
      part[p] = sum;
      __syncthreads();
      if (p == 0) {
         for(i = 0; i < MTCS_BLOCK; ++i)
            scale_updates[filter] += part[i];
      }
   }

   //原型 backward_scale_gpu blas.h blas_ernels.cu
   public$ void   backwardScale(NData *deltaData,NormData *normData){
      //调用方法 backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);
      int w1=deltaData->getWidth();
      int h1=deltaData->getHeight();
      int c1=deltaData->getChannels();

      int w2=normData->getWidth();
      int h2=normData->getHeight();
      int c2=normData->getChannels();
      int batch=normData->getBatch();

      if(w1!=w2 || h1!=h2 || c1!=c2){
         a_error("更新ScalDataUpdates的参数不正确。");
         return;
      }
      int n=c1;
      int spatial=w1*h1;
      //printf("backwardScale --- batch:%d n:%d size:%d c:%d\n",batch,n,size,c1);
      float *x_norm=normData->getXNorm();
      float *deltas=deltaData->getDataArray();

      /* compare
      float *cpuXNorm=malloc(batch*deltaData->getSize()*sizeof(float));
      MtcsMem.memcpy(cpuXNorm,x_norm,batch*deltaData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuDelta=malloc(batch*deltaData->getSize()*sizeof(float));
      MtcsMem.memcpy(cpuDelta,deltas,batch*deltaData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpuUpdates=malloc(size*sizeof(float));
      MtcsMem.memcpy(cpuUpdates,scale_updates,size*sizeof(float),MtcsCpyKind.DEV2HOST);
      int i,b,f;
      for(f = 0; f < n; ++f){
         float sum = 0;
         for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
               int index = i + size*(f + n*b);
               sum += cpuDelta[index] * cpuXNorm[index];
            }
         }
         cpuUpdates[f] += sum;
      }
      */
      backward_scale_kernel<<<n, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(x_norm, deltas, batch, n, spatial, scale_updates);

      /* compare
      float *compareCpu=malloc(size*sizeof(float));
      MtcsMem.memcpy(compareCpu,scale_updates,size*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsScaleData backwardScale 比较\n");
      DnnUtils.compare(size,cpuUpdates,compareCpu);
      free(cpuXNorm);
      free(cpuDelta);
      free(cpuUpdates);
      free(compareCpu);
      */
   }


   ~MtcsScaleData(){
      freeData();
   }
};

