#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include "MtcsIOData.h"
#include "../DnnUtils.h"
#include "../cnnmicro.h"
#include "MtcsTool.h"
#include "MtcsActivation.h"

impl$ MtcsIOData{

   MtcsIOData(int w,int h,int channels,int batch){
      self->w=w;
      self->h=h;
      self->channels=channels;
      self->batch=batch;
      self->size=w*h*channels;
      dataArray = MtcsMem.malloc(batch*size*sizeof(float),TRUE);
      destroyFunc=MtcsMem.free;
   }

   public$ MtcsIOData(int size,int batch){
      self->w=0;
      self->h=0;
      self->channels=0;
      self->batch=batch;
      self->size=size;
      dataArray = MtcsMem.malloc(batch*size*sizeof(float),TRUE);
      destroyFunc=MtcsMem.free;
   }

   /**
    * dataArray的地址指向的是分配的batch*memsize内存。
    */
   void  setZero(){
      MtcsTool.fill/*!fill_ongpu(n, 0, d, 1);*/(batch*size, 0, dataArray);
   }

   /**
    * 实现InputData接口
    */
   public$ void setImageData(float *imageData,int i){
      //printf("mtcsiodata.c setImageData i:%d size:%d dataArray:%p %p\n",i,size,dataArray,&dataArray[i]);
      MtcsMem.memcpy(dataArray+i*size,imageData,size*sizeof(float),MtcsCpyKind.HOST2DEV);
   }

   public$ void   setImageData(float *imageData){
      MtcsMem.memcpyAsync(dataArray,imageData,batch*size*sizeof(float),MtcsCpyKind.HOST2DEV,MtcsTool.getStream());
   }

   __global__ void copy(int size, float *src, float *dst){
       int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index < size)
           dst[index] = src[index];
   }

   /**
    * 覆盖父类NData的copy方法。
    */
   aboolean copy(NData *dest){
      if(!(dest varof$ MtcsIOData)){
         return super$->copy(dest);
      }
      aboolean ret= compare(dest);
      if(dest->readOnly){
         a_error("目标是只读对象。");
         return FALSE;
      }
      if(!ret){
         a_warning("不能复制数据到dest。参数不匹配！");
         return FALSE;
      }
      if(size!=dest->getSize()){
         a_error("目标与源的大小不匹配。dest:%d src:%d",dest->getSize(),size);
         return FALSE;
      }
      const int numBlocks = (batch*size) / MTCS_BLOCK + 1;
      float *destArray=dest->getDataArray();
      printf("destArray----%p\n",destArray);
      copy<<<numBlocks, MTCS_BLOCK, 0,MtcsTool.getStream()>>>(batch*size,dataArray,destArray);
      return TRUE;
   }

   private$ IOData *createCPUActivateData(ActivationType type){
      IOData *dataCPU=new$  IOData(size,batch);
      MtcsMem.memcpy(dataCPU->dataArray,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      dataCPU->activate(type);
      return dataCPU;
   }

   //原型 activate_array_cpu_custom gemm.h gemm.c
   void activate(const ActivationType type){
      int totalSize/*! l.outputs*l.batch*/ = getSize()*batch;
      const int num_blocks = MtcsTool.getNumberOfBlocks/*!get_number_of_blocks*/(totalSize, MTCS_BLOCK);
      /* compare
      IOData *dataCPU=createCPUActivateData(type);
      */

      if (type == ActivationType.LINEAR)
         return;
      else if (type == ActivationType.LEAKY || type == ActivationType.REVLEAKY){
         MtcsActivation.leaky <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      }else if (type == ActivationType.LOGISTIC)
         MtcsActivation.logistic <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else if (type == ActivationType.TANH)
         MtcsActivation.tanh <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else if (type == ActivationType.HARDTAN)
         MtcsActivation.hardtan <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else if (type == ActivationType.RELU)
         MtcsActivation.relu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else if (type == ActivationType.RELU6)
         MtcsActivation.relu6 <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else if (type == ActivationType.SELU)
         MtcsActivation.selu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else if (type == ActivationType.GELU)
         MtcsActivation.gelu <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray,totalSize);
      else{
         MtcsActivation.activateArray<<<MtcsTool.gridSize(totalSize), MTCS_BLOCK, 0, MtcsTool.getStream()>>>
               (dataArray,totalSize,type);
      }
      /* compare
      printf("MtcsIOData.c 激活函数比较。batch:%d dataCPU：%p\n",batch,dataCPU);
      float *compareCpu=malloc(batch*size*sizeof(float));
      MtcsMem.memcpy(compareCpu,dataArray,batch*size*sizeof(float),MtcsCpyKind.DEV2HOST);
      DnnUtils.compare(batch*size,compareCpu,dataCPU->dataArray);
      dataCPU->unref();
      free(compareCpu);
      */
   }

   //原型 activate_array_swish_kernel activation_kernels.cu
   private$ __global__ void activateSwitch(float *x, int n, float *output_sigmoid_gpu, float *output_gpu){
       int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (i < n) {
           float x_val = x[i];
           float sigmoid = MtcsActivation.logistic(x_val);
           if (output_sigmoid_gpu)
              output_sigmoid_gpu[i] = sigmoid;
           output_gpu[i] = x_val * sigmoid;
       }
   }

   //覆盖父类方法
   void activateArraySwish(OutputData * activation_input){
      int totalSize=batch*size;
      float *output_sigmoid_gpu=NULL;
      if(activation_input)
         output_sigmoid_gpu=activation_input->getDataArray();
      activateSwitch <<<MtcsTool.gridSize(totalSize), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
            (dataArray, totalSize, output_sigmoid_gpu,dataArray);
   }

   //原型 scale_bias_kernel blas_kernels.cu
   __global__ void scale_bias_kernel(float *output, float *scale, int batch, int filters, int spatial, int current_size){
       const int index = blockIdx.x*blockDim.x + threadIdx.x;
       if (index >= current_size)
          return;
       int f = (index / spatial) % filters;
       output[index] *= scale[f];
   }

   /**
    * 覆盖父类方法 scale
    */
   void scale(float *scales){
      scaleBias(scales);
   }

   //原型 scale_bias_gpu blas.h blas_kernels.cu
   void scaleBias(float *scale){
       const int current_size = batch * channels * w*h;
       const int num_blocks = MtcsTool.getNumberOfBlocks(current_size, MTCS_BLOCK);
       scale_bias_kernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
             (dataArray, scale, batch, channels, w*h, current_size);
   }

   __global__ void add_bias_kernel(float *output, float *biases, int batch, int filters, int spatial, int current_size){
      const int index = blockIdx.x*blockDim.x + threadIdx.x;
      if (index >= current_size)
         return;

      int f = (index / spatial) % filters;
      output[index] += biases[f];
   }

   //原型 add_bias blas.h convolutional_layer.c
   //覆盖父类方法
   void addBias(float *biases){
      int spatial = w*h;
      const int totalSize = batch * size;
      const int num_blocks = MtcsTool.getNumberOfBlocks(totalSize, MTCS_BLOCK);
      add_bias_kernel <<<num_blocks, MTCS_BLOCK, 0, MtcsTool.getStream() >>>(dataArray, biases, batch, channels, spatial, totalSize);
   }

   ~MtcsIOData(){
      //printf("MtcsIOData --------------------------release %p %d %d %d %d %d\n",self,self->w,self->h,self->channels,self->size,self->batch);
   }
};

