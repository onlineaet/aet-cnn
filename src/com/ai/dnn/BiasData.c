#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/util/ARandom.h>
#include "BiasData.h"
#include "DnnUtils.h"



impl$ BiasData{

   BiasData(int size){
      self->size=size;
      bias=calloc(size,sizeof(float));//偏差初始化为0
      updates=calloc(size,sizeof(float));//偏差的梯度初始化为0
      ema=NULL;
   }

   void freeData(){
      if(bias){
         free(bias);
         bias=NULL;
      }
      if(updates){
         free(updates);
         updates=NULL;
      }
      if(ema){
         free(ema);
         ema=NULL;
      }
   }

   /*
   *计算偏差的梯度
   *求偏差的梯度 ∂C/∂b^lj=δ^lj l是层数，j是第几号偏差
   */
   //原型 backward_bias convolutional_layer.h convolutional_layer.c
   void calcGrad(NData *deltaData){
      int batch=deltaData->getBatch();
      int channels=deltaData->getChannels();
      int sizePerChannel=deltaData->getSize()/channels;
      if(channels!=size){
         a_error("错误的参数:delta的通道数：%d 偏差的大小:%d",size,channels);
         return;
      }
      int b,i;
      //不能用并行，与gpu比较误差大，不用并行后，比较在误差范围内，为什么？
      //#pragma omp parallel for  private(b,i)
      float *delta=deltaData->getDataArray();
      for(b = 0; b < batch; ++b){
         for(i = 0; i < channels; ++i){
            updates[i] += DnnUtils.sum(delta+sizePerChannel*(i+b*channels), sizePerChannel);
         }
      }
   }


   float *getBias(){
      return bias;
   }

   public$ float *getUpdates(){
      return updates;
   }


   public$ void setBiasValue(float value){
      int i;
      for(i=0;i<size;i++)
         bias[i]=value;
   }

   /**
    *把updates加到bias上
    *原型 axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    */
   void   addUpdatesToBias(float alpha){
      int i;
      for(i=0;i<size;i++){
         bias[i]+=alpha*updates[i];
      }
   }

   /**
   * 缩放梯度
   */
   void   scaleUpdates(float scale){
      int i;
      for(i=0;i<size;i++){
         updates[i]*=scale;
      }
   }

   //只有train才需要创建ema数据
   public$ void   createEma(){
      if(!ema)
         ema=calloc(size,sizeof(float));
   }

   public$ void   createUpdates(){
      if(!updates)
         updates=calloc(size,sizeof(float));
   }

   public$ void clear(){
      int i;
      for (i = 0; i < size; ++i)
         bias[i] = 0;
   }

   BiasData *clone(){
      BiasData *newData=new$ BiasData(size);
      memcpy(newData->bias,bias,size*sizeof(float));
      memcpy(newData->updates,updates,size*sizeof(float));
      if(ema){
         newData->createEma();
         memcpy(newData->ema,ema,size*sizeof(float));
      }
      return newData;
   }

   public$ int  getSize(){
      return size;
   }

   public$ int read(FILE *fp){
      return fread(bias,sizeof(float),size,fp);
   }

   public$ float *getEma(){
      return ema;
   }


   ~BiasData(){
      freeData();
   }
};

