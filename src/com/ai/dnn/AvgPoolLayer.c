#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <aet/mtcs/MtcsSystem.h>
#include <aet/mtcs/MtcsMem.h>

#include "NNetwork.h"
#include "AvgPoolLayer.h"
#include "cnnmicro.h"
#include "mtcs/MtcsTool.h"
#include "DnnUtils.h"

impl$ AvgPoolLayer{

   AvgPoolLayer(int batch, int w, int h, int c){
      fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
      self->type = LayerType.AVGPOOL;
      self->batch = batch;
      setInputDimen(w,h,c);
      setOutputDimen(1,1,c);
      self->outputs = 1*1*c;
      self->inputs = h*w*c;
      outputData=DataFactory.getInstance()->createOutputData(outputs,batch);
      deltaData=DataFactory.getInstance()->createDeltaData(outputs,batch);
   }

   void forwardCPU(NetworkState state){
      int b,i,k;
      float *output = outputData->getDataArray();
      float *input = state.input->getDataArray();
      int c = inputDimen.channels;
      int inputWH = inputDimen.w*inputDimen.h;
      for(b = 0; b < batch; ++b){
         for(k = 0; k < c; ++k){
            int out_index = k + b*c;
            output[out_index] = 0;
            for(i = 0; i < inputWH; ++i){
               int in_index = i + inputWH*(k + b*c);
               output[out_index] += input[in_index];
            }
            output[out_index] /= inputWH;
         }
      }
   }

   //原型 forward_avgpool_layer_kernel avgpool_layer_kernels.cu
   __global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output){
      int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if(id >= n)
         return;

      int k = id % c;
      id /= c;
      int b = id;

      int i;
      int out_index = (k + c*b);
      output[out_index] = 0;
      for(i = 0; i < w*h; ++i){
         int in_index = i + h*w*(k + b*c);
         output[out_index] += input[in_index];
      }
      output[out_index] /= w*h;
   }


   /**
   * 计算本层l的误差,需要l+1层的误差。
   * 本层没有权重，所以计算本层误差时，W=1.
   */
   void backwardCPU(NetworkState state){
      int b,i,k;
      int c =inputDimen.channels;
      int inputWH=inputDimen.w*inputDimen.h;
      float *layerDelta=deltaData->getDataArray();
      float *delta=state.delta->getDataArray();
      for(b = 0; b < batch; ++b){
         for(k = 0; k < c; ++k){
            int out_index = k + b*c;
            for(i = 0; i < inputWH; ++i){
               int in_index = i + inputWH*(k + b*c);
               delta[in_index] += layerDelta[out_index] / (inputWH);
            }
         }
      }
   }


   //原型 backward_avgpool_layer_kernel avgpool_layer_kernels.cu
   __global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta){
       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if(id >= n) return;

       int k = id % c;
       id /= c;
       int b = id;

       int i;
       int out_index = (k + c*b);
       for(i = 0; i < w*h; ++i){
           int in_index = i + h*w*(k + b*c);
           in_delta[in_index] += out_delta[out_index] / (w*h);
       }
   }


   void resize(int w, int h){
      setInputDimen(w,h);
      self->inputs = h*w*inputDimen.channels;
   }

    /**
     * 测试mtcs与cpu计算结果是否相同
     */
   float *testforwardCpu(NetworkState state){
      int c =inputDimen.channels;
      int inputWidth=inputDimen.w;
      int inputHeight=inputDimen.h;
      int inputWH=inputHeight*inputWidth;
      float *outputCpu=malloc(outputs*batch*sizeof(float));
      float *inputCpu=malloc(inputs*batch*sizeof(float));
      MtcsMem.memcpy(inputCpu,state.input->getDataArray(),inputs*batch*sizeof(float),MtcsCpyKind.DEV2HOST);
      MtcsMem.memcpy(outputCpu,outputData->getDataArray(),outputs*batch*sizeof(float),MtcsCpyKind.DEV2HOST);

      int b,i,k;
      for(b = 0; b < batch; ++b){
         for(k = 0; k < c; ++k){
            int out_index = k + b*c;
            outputCpu[out_index] = 0;
            for(i = 0; i < inputWH; ++i){
               int in_index = i + inputWH*(k + b*c);
               outputCpu[out_index] += inputCpu[in_index];
            }
            outputCpu[out_index] /= inputWH;
         }
      }

      free(inputCpu);
      return outputCpu;
   }

   void testoutput(float *src,int size,char *explain,aboolean iscpu){
      int n=size>100?100:size;
      float *cpudata = src;
      if(!iscpu){
         cpudata=(float*)malloc(n*sizeof(float));
         MtcsMem.memcpy(cpudata,src,n*sizeof(float),MtcsCpyKind.DEV2HOST);
      }
      int i;
      for(i=0;i<n;i++)
         printf("%s is i:%d %f layer:%d\n",explain,i,cpudata[i],getOrderNumber());
      if(!iscpu)
         free(cpudata);
   }

   void forward(NetworkState state){
      if(state.input==NULL){
         a_error("输入数据是空的。");
      }
      if(devType==DeviceType.CPU){
         forwardCPU(state);
         //testoutput(outputData->getDataArray(),outputData->getSize()*batch,"AvgPoolLayer forward CPU 输出数据",TRUE);
      }else if(devType==DeviceType.MTCS){
         /*compare
         float *outputCpu=testforwardCpu(state);
         */

         size_t n = inputDimen.channels*batch;//layer.c*layer.batch;
         forward_avgpool_layer_kernel<<<MtcsTool.gridSize(n)/*!cuda_gridsize(n)*/, MTCS_BLOCK,
                     0, MtcsTool.getStream()/*!get_cuda_stream()*/>>>(n, inputDimen.w,inputDimen.h,inputDimen.channels,
                           state.input->getDataArray(),outputData->getDataArray());

        /* compare
         float *compareCpu=malloc(outputs*batch*sizeof(float));
         MtcsMem.memcpy(compareCpu,outputData->getDataArray(),outputs*batch*sizeof(float),MtcsCpyKind.DEV2HOST);
         printf("AvgPoolLayer forward 比较\n");
         DnnUtils.compare(batch*outputs,compareCpu,outputCpu);
         free(outputCpu);
         free(compareCpu);
          */


      }else{
         a_error("不支持的设备:%d\n",devType);
      }
   }

   float *testbackwardCPU(NetworkState state){
      int b,i,k;
      int c =inputDimen.channels;
      int inputWH=inputDimen.w*inputDimen.h;
      float *layerDelta=deltaData->getDataArray();
      float *delta=state.delta->getDataArray();

      float *layerDeltaCPU=malloc(deltaData->getSize()*batch*sizeof(float));
      MtcsMem.memcpy(layerDeltaCPU,layerDelta,deltaData->getSize()*batch*sizeof(float),MtcsCpyKind.DEV2HOST);

      float *deltaCpu=malloc(state.delta->getSize()*batch*sizeof(float));
      MtcsMem.memcpy(deltaCpu,delta,state.delta->getSize()*batch*sizeof(float),MtcsCpyKind.DEV2HOST);

      for(b = 0; b < batch; ++b){
         for(k = 0; k < c; ++k){
            int out_index = k + b*c;
            for(i = 0; i < inputWH; ++i){
               int in_index = i + inputWH*(k + b*c);
               deltaCpu[in_index] += layerDeltaCPU[out_index] / (inputWH);
            }
         }
      }
      free(layerDeltaCPU);
      return deltaCpu;
   }

   //本层属于隐藏层 误差公式如下：  δ^l = ((w^l+1 )^T δ^(l+1)) ⊙ σ′(z^l)
   //由于 σ′(z^l)=1,所以本层的误差就是上一层的误差。
   void backward(NetworkState state){
      if(devType==DeviceType.CPU){
         backwardCPU(state);
      }else if(devType==DeviceType.MTCS){
         /*compare
         float *deltaCpu=testbackwardCPU(state);
         */

         size_t n = inputDimen.channels*batch;
         backward_avgpool_layer_kernel<<<MtcsTool.gridSize(n), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
               (n, inputDimen.w, inputDimen.h, inputDimen.channels, state.delta->getDataArray(), deltaData->getDataArray());

         /* compare
         float *compareCpu=malloc(state.delta->getSize()*batch*sizeof(float));
         MtcsMem.memcpy(compareCpu, state.delta->getDataArray(), state.delta->getSize()*batch*sizeof(float),MtcsCpyKind.DEV2HOST);
         printf("AvgPoolLayer backward 比较\n");
         DnnUtils.compare(state.delta->getSize()*batch,deltaCpu,compareCpu);
         free(deltaCpu);
         free(compareCpu);
         */

      }else{
         a_error("不支持的设备:%d\n",devType);
      }
   }

};

