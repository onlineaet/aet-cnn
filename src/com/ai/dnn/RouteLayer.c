#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include "RouteLayer.h"
#include "DnnUtils.h"
#include "NNetwork.h"


/**
 * 将几个输入层拼接在一起，例如：输入层1: 26*26*256   输入层2： 26*26*128， 则route输出为：26*26*(256+128)；
 */

impl$ RouteLayer{

   RouteLayer(int batch, int layerCount,NLayer **inputLayers, int groups, int group_id, apointer network){
      fprintf(stderr,"route ");
      self->type =LayerType.ROUTE;
      self->batch = batch;
      self->layerCount = layerCount;
      self->inputLayers=inputLayers;
      self->groups = groups;
      self->group_id = group_id;
      self->wait_stream_id = -1;
      int i;
      int outputs = 0;
      for(i = 0; i < layerCount; ++i){
         fprintf(stderr," %d", inputLayers[i]->outputs);
         outputs += inputLayers[i]->outputs;
      }
      outputs = outputs / groups;
      self->outputs = outputs;
      self->inputs = outputs;
      //fprintf(stderr, " inputs = %d \t outputs = %d, groups = %d, group_id = %d \n", l.inputs, l.outputs, l.groups, l.group_id);
      //l.delta = (float*)xcalloc(outputs * batch, sizeof(float));
      //l.output = (float*)xcalloc(outputs * batch, sizeof(float));
      outputData=DataFactory.getInstance()->createOutputData(outputs,batch);
      deltaData=DataFactory.getInstance()->createDeltaData(outputs,batch);
      setNetwork(network);
      #ifdef GPU
      l.forward_gpu = forward_route_layer_gpu;
      l.backward_gpu = backward_route_layer_gpu;

      l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
      l.output_gpu = cuda_make_array(l.output, outputs*batch);
      #endif
   }

   //原型 forward_route_layer route_layer.c

   void forward(NetworkState state){
      NNetwork *net=(NNetwork *)network;
      int i, j;
      int offset = 0;
      for(i = 0; i < self->layerCount; ++i){
         NLayer *layer=inputLayers[i];
         OutputData *layerOutput=layer->getOutputData();
         int layerOutputSize = ((IOData*)layerOutput)->getSize();
         for(j = 0; j < self->batch; ++j){
            ((IOData*)layerOutput)->copy((IOData*)outputData,j,offset);//把layerOutput中的数据复制到outputData中
         }
         offset += layerOutputSize;
      }
   }

   /**
   * 根据误差公式
   *  δ^l = ((w^(l+1) )^T δ^(l+1)) ⊙ σ′(z^l)
   *  通过hadamard乘积 本层的激活函数导数值,
   *  该方法由下一层调用。但本层的误差不需要，所以是空的。
   */
   void multActivationFuncDerivative(){
   }


   void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
   {
       int i;
       for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
   }

   void backward(NetworkState state){
      int i, j,z;
      int offset = 0;
      for(i = 0; i < layerCount; ++i){
         NLayer *layer=inputLayers[i];
         DeltaData *layerDeltaData = layer->deltaData;/*!state.net.layers[index].delta;*/
         int input_size = layer->inputs/*!l.input_sizes[i]*/;
         int part_input_size = input_size / groups;
         for(j = 0; j < batch; ++j){
            float *delta = deltaData->getData(j)+part_input_size;
            float *layerDelta = layerDeltaData->getData(j)+part_input_size;
            /*!axpy_cpu(part_input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size + part_input_size*l.group_id, 1);*/
            for(z=0;z<part_input_size;++z)
               layerDelta[z]+=delta[z];
         }
      }
   }

   /**
   * 有关联的层的输出大小来更新本层的输出大小
   */
   void updataDimension(){
      NNetwork *net=(NNetwork *)network;
      NLayer *first=inputLayers[0];
      setOutputDimen(first->outputDimen.w,first->outputDimen.h,first->outputDimen.channels);
      int i;
      for(i = 1; i < layerCount; ++i){
         NLayer *next=inputLayers[i];
         if(next->outputDimen.w== first->outputDimen.w && next->outputDimen.h == first->outputDimen.h){
            outputDimen.channels += next->outputDimen.channels;
         }else{
            fprintf(stderr, " The width and height of the input layers are different. \n");
            setOutputDimen(0,0,0);
         }
      }
      // printf("route out --- %d %d %d %d %d\n",out_w,out_h,out_c,outputs,out_w*out_h*out_c);
      //layer.out_c = layer.out_c / layer.groups;
      self->outputDimen.channels = self->outputDimen.channels / self->groups;
      setInputDimen(first->inputDimen.w,first->inputDimen.h,self->outputDimen.channels);
   }

   void resize(int w, int h){
      NNetwork *net=(NNetwork *)network;
      int i;
      printf("route resize: 层:%d input_layers[0]:%d outputData:%p w:%d h:%d old_w:d old_h:%d\n",
               getOrderNumber(),self->input_layers[0],outputData,w,h,inputDimen.w,inputDimen.h);
      int outputs=getLayerOutputSize();
      self->outputs = outputs;
      self->inputs = outputs;
      ((IOData*)outputData)->resize(outputs);
      deltaData->resize(outputs);
      updataDimension();
   }

   int getLayerOutputSize(){
      NNetwork *net=(NNetwork *)network;
      int i;
      int outputs=0;
      for(i=0;i<self->layerCount;i++){
         int index=input_layers[i];
         NLayer *layer=net->getLayer(index);
         OutputData *outputData=layer->getOutputData();
         if(outputData){
            outputs+=((IOData*)outputData)->getSize();
         }else{
            a_error("不应该是空的outputdata\n");
         }
      }
      return outputs;
   }

   void setReceptive(int *w,int *h,int *wScale,int *hScale){
      *w = *h = *wScale = *hScale = 0;
      NNetwork *net=(NNetwork*)network;
      int k;
      for (k = 0; k < layerCount; ++k) {
         NLayer *layer = net->getLayer(self->input_layers[k]);
         *w = max_val_cmp(*w, layer->receptive.w);
         *h = max_val_cmp(*h, layer->receptive.h);
         *wScale = max_val_cmp(*wScale, layer->receptive.wScale);
         *hScale = max_val_cmp(*hScale, layer->receptive.hScale);
      }
      self->receptive.w = *w;
      self->receptive.h = *h;
      self->receptive.wScale = *wScale;
      self->receptive.hScale = *hScale;
   }
};

