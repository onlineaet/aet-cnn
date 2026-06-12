#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/util/ARandom.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include "MtcsBiasData.h"
#include "MtcsTool.h"
#include "../DnnUtils.h"


impl$ MtcsBiasData{

   MtcsBiasData(int size){
      self->size=size;
      bias=MtcsMem.calloc(size*sizeof(float),TRUE);//偏差初始化为0
      updates=MtcsMem.calloc(size*sizeof(float),TRUE);//偏差的梯度初始化为0
      ema=NULL;
   }

   /**
    * 计算每个输出的偏差，一个输出特征图的通道数等于卷积核的个数，
    * n=卷积核个数 size=输出的特征图的w*h
    */
   //原型 backward_bias_kernel blas_kernels.cu
   __global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size){
       __shared__ float part[MTCS_BLOCK];
       int i,b;
       int filter = blockIdx.x;
       int p = threadIdx.x;
       float sum = 0;
       for(b = 0; b < batch; ++b){
           for(i = 0; i < size; i += MTCS_BLOCK){
               int index = p + i + size*(filter + n*b);
               sum += (p+i < size) ? delta[index] : 0;
           }
       }
       part[p] = sum;
       __syncthreads();
       if (p == 0) {
           for(i = 0; i < MTCS_BLOCK; ++i)
              bias_updates[filter] += part[i];
       }
   }

   void calcGradCpu(float *cpuUpdates,float *delta,int batch,int channels,int sizePerChannel){
      int b,i;
      //不能用并行，与gpu比较误差大，不用并行后，比较在误差范围内，为什么？
      //#pragma omp parallel for  private(b,i)
      for(b = 0; b < batch; ++b){
         for(i = 0; i < channels; ++i){
            cpuUpdates[i] += DnnUtils.sum(delta+sizePerChannel*(i+b*channels), sizePerChannel);
         }
      }
   }

   /*
   *计算偏差的梯度
   *求偏差的梯度 ∂C/∂b^lj=δ^lj l是层数，j是第几号偏差
   *覆盖父类的calcGrad方法
   */
   void calcGrad(NData *deltaData){
      int batch=deltaData->getBatch();
      int channels=deltaData->getChannels();
      int sizePerChannel=deltaData->getSize()/channels;
      if(channels!=size){
         a_error("错误的参数:delta的通道数：%d 偏差的大小:%d",size,channels);
         return;
      }
      /* compare
      float *cpuUpdates=malloc(size*sizeof(float));
      MtcsMem.memcpy(cpuUpdates,updates,size*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *cpudelta=malloc(batch*deltaData->getSize()*sizeof(float));
      MtcsMem.memcpy(cpudelta,deltaData->getDataArray(),batch*deltaData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
      calcGradCpu(cpuUpdates,cpudelta,batch,channels,sizePerChannel);
      */
      backward_bias_kernel<<<channels, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(updates,
            deltaData->getDataArray(), batch, channels, sizePerChannel);

      /*compare
      float *compareUpdates=malloc(size*sizeof(float));
      MtcsMem.memcpy(compareUpdates,updates,size*sizeof(float),MtcsCpyKind.DEV2HOST);
      printf("MtcsBiasData calcGrad 比较\n");
      DnnUtils.compare(size,compareUpdates,cpuUpdates);
      free(cpuUpdates);
      free(cpudelta);
      free(compareUpdates);
      */
   }

   public$ void   createEma(){
      if(!ema)
         ema=MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   public$ void   createUpdates(){
      if(!updates)
         updates=MtcsMem.calloc(size*sizeof(float),TRUE);
   }

   void freeData(){
      if(bias){
         MtcsMem.free(bias);
         bias=NULL;
      }
      if(updates){
         MtcsMem.free(updates);
         updates=NULL;
      }
      if(ema){
         MtcsMem.free(ema);
         ema=NULL;
      }
   }

   ~MtcsBiasData(){
      freeData();
   }
};

