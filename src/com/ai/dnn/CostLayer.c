#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include "CostLayer.h"
#include "DnnUtils.h"
#include "NNetwork.h"
#include "cnnmicro.h"

impl$ CostLayer{

   CostLayer(int batch, int inputs, CostType costType, float scale,float ratio){
      fprintf(stderr, "cost                                           %4d\n",  inputs);
      self->type = LayerType.COST;
      self->scale = scale;
      self->ratio = ratio;
      self->batch = batch;
      self->inputs = inputs;
      self->outputs = inputs;
      self->costType = costType;
      deltaData=DataFactory.getInstance()->createDeltaData(inputs,batch);
      outputData=DataFactory.getInstance()->createOutputData(inputs,batch);
      self->cost=0;
   #ifdef GPU
      l.forward_gpu = forward_cost_layer_gpu;
      l.backward_gpu = backward_cost_layer_gpu;

      l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
      l.output_gpu = cuda_make_array(l.output, inputs*batch);
   #endif
   }

   /*
   原型 l2_cpu(int n, float *pred, float *truth, float *delta, float *error) blas.h blas.c
   均方差 SSE 损失函数如下。
   损失公式：L = 1/2 * ∑(y_i - ŷ_i)² 注意有的定为 真实值-计算值 即 (ŷ_i - y_i)
   y_i：真实标签（one-hot 编码）
   ŷ_i：模型预测值（softmax 输出）
   求输出层误差的公式如下:注意特指输出层，隐藏层的误差依赖上一层。
   δ =(∂C/ ∂a)*σ′ (z j )
   用的是线性激活函数 f(x)=x
   因为σ (z j )=zj
   所以 σ′ (z j )=1
   (∂C/∂a^L)偏导数据公式等于(y_i - ŷ_i)
   */
   void l2Cpu(NetworkState state){
      /*!
      int i;
      for(i = 0; i < n; ++i){
      float diff = truth[i] - pred[i];
      error[i] = diff * diff;
      delta[i] = diff;
      }
      */

      int b,j;
      for(b=0;b<batch;++b){
         float *pred=state.input->getData(b);
         float *truth=state.truth->truths[b].values[0];//对于分类 truth 是one-hot编码的向量,如{0,0,1,0,0,0}
         float *delta=deltaData->getData(b);
         float *error=outputData->getData(b);
         for(j=0;j<inputs;j++){
            float diff = truth[j] - pred[j];
            error[j] = diff * diff; // 平方误差 损失函数的公式是  L = 1/2 * ∑(y_i - ŷ_i)² error是这部份(y_i - ŷ_i)²的值。
            delta[j] = diff; // 误差 (∂C/∂a) σ′(z)
         }
      }
   }

   //原型 smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error) blash.h blash.c
   void smoothL1Cpu(NetworkState state){
      /*!
      int i;
      for(i = 0; i < n; ++i){
      float diff = truth[i] - pred[i];
      float abs_val = fabs(diff);
      if(abs_val < 1) {
      error[i] = diff * diff;
      delta[i] = diff;
      }
      else {
      error[i] = 2*abs_val - 1;
      delta[i] = (diff < 0) ? 1 : -1;
      }
      }
      */
      int b,j;
      for(b=0;b<batch;++b){
         float *pred=state.input->getData(b);
         float *truth=state.truth->truths[b].values[0];
         float *delta=deltaData->getData(b);
         float *error=outputData->getData(b);
         for(j=0;j<inputs;j++){
            float diff = truth[j] - pred[j];
            float abs_val = fabs(diff);
            if(abs_val < 1) {
               error[j] = diff * diff;
               delta[j] = diff;
            }else {
               error[j] = 2*abs_val - 1;
               delta[j] = (diff < 0) ? 1 : -1;
            }
         }
      }
   }


   //原型 forward_route_layer route_layer.c
   void forward(NetworkState state){
      NNetwork *net=(NNetwork *)network;
      if (!state.truth)
         return;
      if(costType == MASKED){
         int i,j;
         for(i = 0; i < batch; ++i){
            float *input=state.input->getData(i);
            for(j=0;j<inputs;j++)
               if(state.truth->truths[i].values[0][j]/*!truthstate.truth[i]*/ == SECRET_NUM)
                  input[j]/*!state.input[i]*/ = SECRET_NUM;
         }
      }
      if(costType == SMOOTH){
         smoothL1Cpu/*!smooth_l1_cpu*/(state);
      } else {
         l2Cpu/*!l2_cpu*/(state);
      }
      cost = outputData->sum();/*!sum_array(l.output, l.batch*l.inputs);*/
   }

   //上一层(隐藏层)的误差公式是δ^l = ((w^l+1 )^T δ^(l+1)) ⊙ σ′(z^l)
   //总是乘与它的下一层的误差，所以在这里把误差加入到上一层的误差中。这样上一层只用计算本层的σ′(z^l)
   void backward(NetworkState state){
      //axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, state.delta, 1);
      int b,i;
      for(b=0;b<batch;++b){
         float *delta=deltaData->getData(b);
         float *delta1=state.delta->getData(b);
         for(i=0;i<inputs;++i)
            delta1[i]+=scale*delta[i];
      }
   }

   void resize(int inputs){
   //       l->inputs = inputs;
   //       l->outputs = inputs;
   //       l->delta = (float*)xrealloc(l->delta, inputs * l->batch * sizeof(float));
   //       l->output = (float*)xrealloc(l->output, inputs * l->batch * sizeof(float));
   #ifdef GPU
      cuda_free(l->delta_gpu);
      cuda_free(l->output_gpu);
      l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
      l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
   #endif
   }

   /**
    * 实现 OutputLayer 接口方法
    */
   float getCost(){
      return cost;
   }

};

