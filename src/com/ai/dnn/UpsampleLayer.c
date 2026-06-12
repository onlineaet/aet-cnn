#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/lang/AAssert.h>
#include "UpsampleLayer.h"
#include "DnnUtils.h"
#include "NNetwork.h"

/**
 * 上采样 放大图像
 * 小采样 缩小图像
 * 算法有: 最近邻插值，双线性插值，均值插值，中值插值
 * 这里用的是: 最近邻插值法
 */

impl$ UpsampleLayer{

   UpsampleLayer(int batch, int w, int h, int c, int stride,float scale){
      self->type = LayerType.UPSAMPLE;
      self->batch = batch;
      setInputDimen(w,h,c);
      int out_w = w*stride;
      int out_h = h*stride;
      self->scale=scale;
      self->downSample=FALSE;
      if(stride < 0){
         self->stride = -stride;
         self->downSample=TRUE;
         out_w = w/stride;
         out_h = h/stride;
      }
      setOutputDimen(out_w,out_h,c);
      self->stride = stride;
      self->outputs = out_w*out_h*c;
      self->inputs = w*h*c;
      outputData=DataFactory.getInstance()->createOutputData(out_w,out_h,c,batch);
      deltaData=DataFactory.getInstance()->createDeltaData(out_w,out_h,c,batch);
      if(self->downSample)
         fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, out_w, out_h, c);
      else
         fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, out_w, out_h, c);
   }


   //原型 upsample_cpu blas.h blas.c
   void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out){
      int i, j, k, b;
      for(b = 0; b < batch; ++b){
         for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
               for(i = 0; i < w*stride; ++i){
                  int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                  int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                  if(forward)
                     out[out_index] = scale*in[in_index];
                  else
                     in[in_index] += scale*out[out_index];
               }
            }
         }
      }
   }

   void forward(NetworkState state){
      ((IOData *)outputData)->setZero();
      int out_w=outputDimen.w;
      int out_h=outputDimen.h;
      int c = inputDimen.channels;
      int w =inputDimen.w;
      int h =inputDimen.h;
      float *output=outputData->getDataArray();
      float *input=state.input->getDataArray();

      if(self->downSample){
         upsample_cpu(output, out_w, out_h, c, batch, stride, 0, scale, input);
      }else{
         upsample_cpu(input, w, h, c, batch, stride, 1, scale,output);
      }
   }


   void multActivationFuncDerivative(){
   }




   void backward(NetworkState state){
       int out_w=outputDimen.w;
       int out_h=outputDimen.h;
       int c = inputDimen.channels;
       int w =inputDimen.w;
       int h =inputDimen.h;
       float *output=deltaData->getDataArray();
       float *input=state.delta->getDataArray();

       if(self->downSample){
          upsample_cpu(output, out_w, out_h, c, batch, stride, 1, scale, input);
       }else{
          upsample_cpu(input, w, h, c, batch, stride, 0, scale,output);
       }
    }


   void resize(int w, int h){
      NNetwork *net=(NNetwork *)network;
      setInputDimen(w,h);
      int out_w = w*self->stride;
      int out_h = h*self->stride;
      if(self->downSample){
         out_w = w/self->stride;
         out_h = h/self->stride;
      }
      setOutputDimen(out_w,out_h);
      self->outputs = out_w*out_h*outputDimen.channels;
      self->inputs = h*w*inputDimen.channels;
      outputData->resize(out_w,out_h);
      deltaData->resize(out_w,out_h);
   }

   //
   void setReceptive(int *w,int *h,int *wScale,int *hScale){
      int stride = max_val_cmp(1, self->stride);

      self->receptive.w = *w;
      self->receptive.h = *h;
      self->receptive.wScale = *wScale=*wScale/stride;
      self->receptive.hScale = *hScale=*hScale/stride;
   }

};

