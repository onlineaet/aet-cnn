#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <aet/lang/System.h>
#include <aet/mtcs/MtcsSystem.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/lang/AAssert.h>

#include "MtcsMaxPoolLayer.h"
#include "../DnnUtils.h"
#include "../Activation.h"
#include "../DstbCompute.h"
#include "../DataFactory.h"
#include "../cnnmicro.h"
#include "MtcsTool.h"


impl$ MtcsMaxPoolLayer{

   MtcsMaxPoolLayer (int batch, int h, int w, int c, int size, int stride_x, int stride_y,
               int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train){
        super$(batch,h,w,c,size,stride_x,stride_y,padding,maxpool_depth,out_channels, antialiasing,avgpool,train);
        if (antialiasing)
           input_antialiasing = DataFactory.getInstance()->createInputData(outputDimen.w,outputDimen.h,outputDimen.channels,batch);
   }

   __global__ void forward_maxpool_depth_layer_kernel(int n, int w, int h,
         int c, int out_c, int batch, float *input, float *output, int *indexes){
       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (id >= n)
          return;

       int j = id % w;
       id = id / w;
       int i = id % h;
       id = id / h;
       int b = id % batch;

       int k;
       for (int g = 0; g < out_c; ++g){
           int out_index = j + w*(i + h*(g + out_c*b));
           float max = -FLT_MAX;
           int max_i = -1;

           for (k = g; k < c; k += out_c){
               int in_index = j + w*(i + h*(k + c*b));
               float val = input[in_index];

               max_i = (val > max) ? in_index : max_i;
               max = (val > max) ? val : max;
           }
           output[out_index] = max;
           if (indexes)
              indexes[out_index] = max_i;
       }
   }

   __global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x,
         int stride_y, int size, int pad, float *input, float *output, int *indexes){
       int h = (in_h + pad - size) / stride_y + 1;
       int w = (in_w + pad - size) / stride_x + 1;
       int c = in_c;

       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if(id >= n)
          return;

       int j = id % w;
       id /= w;
       int i = id % h;
       id /= h;
       int k = id % c;
       id /= c;
       int b = id;

       int w_offset = -pad / 2;
       int h_offset = -pad / 2;

       int out_index = j + w*(i + h*(k + c*b));
       float max = -INFINITY;
       int max_i = -1;
       int l, m;
       for(l = 0; l < size; ++l){
           for(m = 0; m < size; ++m){
               int cur_h = h_offset + i*stride_y + l;
               int cur_w = w_offset + j*stride_x + m;
               int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
               int valid = (cur_h >= 0 && cur_h < in_h &&
                       cur_w >= 0 && cur_w < in_w);
               float val = (valid != 0) ? input[index] : -INFINITY;
               max_i = (val > max) ? index : max_i;
               max   = (val > max) ? val   : max;
           }
       }
       output[out_index] = max;
       if (indexes)
          indexes[out_index] = max_i;
   }

   __global__ void forward_zero_nonmax_kernel(int totalSize,int unitSize, float *inputs, float *outputs){
      int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (id >= totalSize)
         return;
      if (inputs[id] != outputs[id])
         outputs[id] = 0;
   }

   void testForwardCPU(float *inputCpu,float *outputCpu,int *indexsCpu){
      int lw = inputDimen.w;
      int lh = inputDimen.h;
      int lc=inputDimen.channels;
      int out_c = outputDimen.channels;

      int b, i, j, k, m, n;
      int w_offset = -pad / 2;
      int h_offset = -pad / 2;

      int h = outputDimen.h;
      int w = outputDimen.w;
      int c = outputDimen.channels;

      for (b = 0; b < batch; ++b) {
         for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
               for (j = 0; j < w; ++j) {
                  int out_index = j + w*(i + h*(k + c*b));
                  float max = -FLT_MAX;
                  int max_i = -1;
                  for (n = 0; n < size; ++n) {
                     for (m = 0; m < size; ++m) {
                        int cur_h = h_offset + i*stride_y + n;
                        int cur_w = w_offset + j*stride_x + m;
                        int index = cur_w + lw*(cur_h + lh*(k + b*lc));
                        int valid = (cur_h >= 0 && cur_h < lh &&
                        cur_w >= 0 && cur_w < lw);
                        float val = (valid != 0) ? inputCpu[index] : -FLT_MAX;
                        max_i = (val > max) ? index : max_i;
                        max = (val > max) ? val : max;
                     }
                  }
                  outputCpu[out_index] = max;
                  if (indexsCpu)
                     indexsCpu[out_index] = max_i;
               }
            }
         }
      }
   }

   void forwardMtcsMaxPool(NetworkState state){
      if (maxpool_depth) {
         int h = outputDimen.h;
         int w = outputDimen.w;
         int c = 1;// layer.out_c;

         size_t n = h*w*c*batch;
         float *inputs=state.input->getDataArray();
         float *outputs=outputData->getDataArray();

         forward_maxpool_depth_layer_kernel <<<MtcsTool.gridSize(n), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
               (n, inputDimen.w, inputDimen.h, inputDimen.channels, outputDimen.channels,batch, inputs, outputs, indexes);
         return;
      }
      /* compare
      float *inputCpu=malloc(batch*inputs*sizeof(float));
      a_assert(inputs==state.input->getSize());
      MtcsMem.memcpy(inputCpu,state.input->getDataArray(),batch*state.input->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *outputCpu=malloc(batch*outputs*sizeof(float));
      a_assert(outputs==outputData->getSize());
      MtcsMem.memcpy(outputCpu,outputData->getDataArray(),batch*outputData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
      int *indexsCpu=malloc(batch*outputs*sizeof(int));
      MtcsMem.memcpy(indexsCpu,indexes,batch*outputs*sizeof(int),MtcsCpyKind.DEV2HOST);
      testForwardCPU(inputCpu,outputCpu,indexsCpu);
      */
      int h = outputDimen.h;
      int w = outputDimen.w;
      int c = outputDimen.channels;
      size_t totalSize = h*w*c*batch;
      float *inputsData=state.input->getDataArray();
      float *outputsData=outputData->getDataArray();
      forward_maxpool_layer_kernel <<<MtcsTool.gridSize(totalSize)/*!cuda_gridsize(n)*/, MTCS_BLOCK, 0, MtcsTool.getStream() >>>
            (totalSize, inputDimen.h, inputDimen.w, inputDimen.channels, stride_x, stride_y, size, pad,
            inputsData, outputsData, indexes);

      /* compare
      float *compareCpu=malloc(batch*outputs*sizeof(float));
      MtcsMem.memcpy(compareCpu,outputData->getDataArray(),batch*outputData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
      float *compareIndexsCpu=malloc(batch*outputs*sizeof(int));
      MtcsMem.memcpy(compareIndexsCpu,indexes,batch*outputData->getSize()*sizeof(int),MtcsCpyKind.DEV2HOST);
      printf("MtcsMaxPoolLayer forward 比较CPU与GPU的outputData \n");
      DnnUtils.compare(batch*outputs,compareCpu,outputCpu);
      DnnUtils.compare(batch*outputs,compareIndexsCpu,indexsCpu);
      free(compareIndexsCpu);
      free(compareCpu);
      free(inputCpu);
      free(outputCpu);
      free(indexsCpu);
      */

      if (maxpool_zero_nonmax) {
         forward_zero_nonmax_kernel <<<MtcsTool.gridSize(totalSize)/*!cuda_gridsize(n)*/, MTCS_BLOCK, 0,  MtcsTool.getStream() >>>
         (totalSize, size,state.input->getDataArray(), outputData->getDataArray());
      }

      if (antialiasing) {
         NetworkState s = { 0 };
         s.train = state.train;
         s.workspace = state.workspace;
         if (!state.train)
            s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )

         OutputData *out=getOutputData();
         s.input = (InputData*)out;

         input_layer->forward(s);

         MtcsTool.simpleCopy(outputs*batch,outputData->getDataArray(),input_antialiasing->getDataArray());
         MtcsTool.simpleCopy/*!simple_copy_kernel*/(input_layer->outputs*input_layer->batch,
               input_layer->outputData->getDataArray(),outputData->getDataArray());
         // simple_copy_ongpu(layer.outputs*layer.batch, layer.output_gpu, layer.input_antialiasing_gpu);
         // simple_copy_ongpu(layer.input_layer->outputs*layer.input_layer->batch, layer.input_layer->output_gpu, layer.output_gpu);
      }
   }

   __global__ void forward_local_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c,
      int stride_x, int stride_y, int size, int pad, float *input, float *output){
      int h = (in_h + pad - size) / stride_y + 1;
      int w = (in_w + pad - size) / stride_x + 1;
      int c = in_c;

      int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (id >= n)
       return;

      int j = id % w;
      id /= w;
      int i = id % h;
      id /= h;
      int k = id % c;
      id /= c;
      int b = id;

      int w_offset = -pad / 2;
      int h_offset = -pad / 2;

      int out_index = j + w*(i + h*(k + c*b));
      float avg = 0;
      int counter = 0;
      int l, m;
      for (l = 0; l < size; ++l) {
         for (m = 0; m < size; ++m) {
            int cur_h = h_offset + i*stride_y + l;
            int cur_w = w_offset + j*stride_x + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
            cur_w >= 0 && cur_w < in_w);
            if (valid) {
               counter++;
               avg += input[index];
            }
         }
      }
      output[out_index] = avg / counter;  // as CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
   }

   void forwardMtcsAvgPool(NetworkState state){
      int h = outputDimen.h;
      int w = outputDimen.w;
      int c = outputDimen.channels;
      size_t n = h*w*c*batch;
      float *input=state.input->getDataArray();
      float *output=outputData->getDataArray();
      forward_local_avgpool_layer_kernel  <<<MtcsTool.gridSize(n), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
            (n, inputDimen.h, inputDimen.h, inputDimen.channels, stride_x, stride_y, size,pad, input, output);
   }

   void forward(NetworkState state){
      if(avgpool){
         forwardMtcsAvgPool(state);
      }else{
         forwardMtcsMaxPool(state);
      }
   }

   __global__ void backward_maxpool_depth_layer_kernel(int totalSize,float *delta, float *prev_delta, int *indexes){
       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if (id >= totalSize)
          return;
       int index = indexes[id];
       prev_delta[index] += delta[id];
   }

   __global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c,
         int stride_x, int stride_y, int size, int pad, float *delta, float *prev_delta, int *indexes){
       int h = (in_h + pad - size) / stride_y + 1;
       int w = (in_w + pad - size) / stride_x + 1;
       int c = in_c;
       int area_x = (size - 1) / stride_x;
       int area_y = (size - 1) / stride_y;

       int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
       if(id >= n)
          return;

       int index = id;
       int j = id % in_w;
       id /= in_w;
       int i = id % in_h;
       id /= in_h;
       int k = id % in_c;
       id /= in_c;
       int b = id;

       int w_offset = -pad / 2;
       int h_offset = -pad / 2;

       float d = 0;
       int l, m;
       for(l = -area_y; l < area_y+1; ++l){
           for(m = -area_x; m < area_x+1; ++m){
               int out_w = (j-w_offset)/stride_x + m;
               int out_h = (i-h_offset)/stride_y + l;
               int out_index = out_w + w*(out_h + h*(k + c*b));
               int valid = (out_w >= 0 && out_w < w &&
                        out_h >= 0 && out_h < h);
               d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
           }
       }
       prev_delta[index] += d;
   }

   __global__ void backward_zero_nonmax_kernel(int n, int *indexes, float *prev_delta){

      int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
      if (id >= n)
         return;
      if (indexes[id] != id)
      prev_delta[id] = 0;
   }

   //原型 backward_maxpool_layer_gpu maxpool_lyaer_kernels.cu
   void backward(NetworkState state){
       if (antialiasing) {
          NetworkState s = { 0 };
          s.train = state.train;
           s.workspace = state.workspace;
           s.delta =deltaData;  // s.delta will be returned to l.delta_gpu
           s.input = input_antialiasing/*!layer.input_antialiasing_gpu*/;
           /*!simple_copy_ongpu(layer.input_layer->outputs*layer.input_layer->batch, layer.delta_gpu, layer.input_layer->delta_gpu);*/
           int totalSize = input_layer->outputs*input_layer->batch;
           MtcsTool.simpleCopy/*!simple_copy_kernel*/(totalSize,deltaData->getDataArray(),input_layer->deltaData->getDataArray());
           /*!backward_convolutional_layer_gpu(*(layer.input_layer), s);*/
           input_layer->backward(s);
       }

       if (maxpool_depth) {
           int h = outputDimen.h;
           int w = outputDimen.w;
           int c = outputDimen.channels;

           size_t totalSize = h * w * c * batch;
           backward_maxpool_depth_layer_kernel <<<MtcsTool.gridSize(totalSize), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
                 (totalSize, deltaData->getDataArray(), state.delta->getDataArray(), indexes);
           return;
       }
       /* compare
       float *myDeltaCpu=malloc(batch*deltaData->getSize()*sizeof(float));
       MtcsMem.memcpy(myDeltaCpu,deltaData->getDataArray(),batch*deltaData->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
       float *preDelta=malloc(batch*state.delta->getSize()*sizeof(float));
       MtcsMem.memcpy(preDelta,state.delta->getDataArray(),batch*state.delta->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
       int *cpuIndexs=malloc(batch*outputs*sizeof(int));
       MtcsMem.memcpy(cpuIndexs,indexes,batch*outputs*sizeof(int),MtcsCpyKind.DEV2HOST);
       int i;
       int n=outputDimen.h*outputDimen.w*outputDimen.channels*batch;
       for(i = 0; i < n; ++i){
          int index = cpuIndexs[i];
          preDelta[index] += myDeltaCpu[i];
       }
       */

       int totalSize =  inputDimen.h*inputDimen.w*inputDimen.channels*batch;
       backward_maxpool_layer_kernel<<<MtcsTool.gridSize(totalSize), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
             (totalSize,inputDimen.h, inputDimen.w, inputDimen.channels, stride_x, stride_y,
                   size, pad, deltaData->getDataArray(), state.delta->getDataArray(), indexes);

       /* compare
       float *compareCpu=malloc(batch*state.delta->getSize()*sizeof(float));
       MtcsMem.memcpy(compareCpu,state.delta->getDataArray(),batch*state.delta->getSize()*sizeof(float),MtcsCpyKind.DEV2HOST);
       printf("MtcsMaxPoolLayer.c backward 比较delta layer:%d\n",getOrderNumber());
       DnnUtils.compare(batch*state.delta->getSize(),compareCpu,preDelta);
       free(compareCpu);
       free(cpuIndexs);
       free(preDelta);
       free(myDeltaCpu);
       */

       if (maxpool_zero_nonmax) {
           backward_zero_nonmax_kernel <<<MtcsTool.gridSize(totalSize), MTCS_BLOCK, 0, MtcsTool.getStream() >>>
                 (totalSize, indexes, state.delta->getDataArray());
       }
   }

};

