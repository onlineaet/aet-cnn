#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "WeightData.h"
#include "DnnUtils.h"

impl$ WeightData {

   WeightData(int size){
      self->size=size;
      weights = calloc(size, sizeof(float));
      weight_updates =NULL;//只有train=true才需要weight_updates数据  calloc(size, sizeof(float));
      ema=NULL;
   }

   int getSize(){
      return size;
   }

   float *getWeights(){
      return weights;
   }

   float *getUpdates(){
      return weight_updates;
   }

   public$ float *getEma(){
      return ema;
   }

   public$ float *getDeform(){
      return weight_deform;
   }

   public$ void   setUpdates(float *newUpdatas){
      self->weight_updates = newUpdatas;
   }


   /**
   *把weight_updates加到weights上
   *原型 axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
   */
   void   addUpdatesToWeight(float alpha){
      int i;
      for(i=0;i<size;i++){
         weights[i]+=alpha*weight_updates[i];
      }
   }

   /**
   *把weights加到weight_updates上
   *原型 axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
   */
   void addWeightToUpdates(float alpha){
      int i;
      for(i=0;i<size;i++){
         weight_updates[i]+=alpha*weights[i];
      }
   }

   public$ void   setValue(float value){
      int i;
      for(i=0;i<size;++i)
         weights[i] = value;
   }

   /**
   * 缩放updates
   * 原型  scal_cpu(l.nweights, momentum, l.weight_updates, 1);
   */
   void   scaleUpdates(float scale){
      int i;
      for(i=0;i<size;i++){
         weight_updates[i]*=scale;
      }
   }

   void checkNan(char *info){
      if(5>3)
         return;
      int n;
      for(n = 0; n < size; ++n){
         if(isnan(weight_updates[n])|| isinf(weight_updates[n])){
            printf("%s weight_updates n:%d isnan %f\n",info,n,weight_updates[n]);
            exit(0);
         }
      }

      for(n = 0; n < size; ++n){
         if(isnan(weights[n])||isinf(weights[n])){
            printf("%s weights n:%d isnan %f\n",info,n,weights[n]);
            exit(0);
         }
      }

   }

   //只有train才需要创建ema数据
   public$ void   createEma(){
      if(!ema)
         ema=calloc(size,sizeof(float));
   }

   public$ void   createUpdates(){
      if(!weight_updates)
         weight_updates=calloc(size,sizeof(float));
   }

   public$ int read(FILE *fp){
      return fread(weights,sizeof(float),size,fp);
   }



   ~WeightData(){

   }
};

impl$ ConvKernel{

   ConvKernel(int ksize,int channels,int filters,int pad,int stride){
      super$(channels*filters*ksize*ksize);
      self->ksize=ksize;
      self->channels=channels;
      self->pad=pad;
      self->stride=stride;
      self->filters=filters;
      //        int len=channels*filters*size*size;
      //        float scale = sqrt(2./(size*size*channels));
      //        float *weights=getWeights();
      //        int i;
      //        for(i = 0; i < len; ++i) //生成随机权重
      //           weights[i] = scale*gaussRand.nextGaussian();
      //        rot180Data=malloc(channels*filters*size*size*sizeof(float));
   }

   public$ ConvKernel(int nweights){
      super$(nweights);
   }

   void testprint(){
      int i;
      int nweights=channels*filters*ksize*ksize;
      float *weights=getWeights();
      for(i=0;i<nweights;i++){
        printf("权重数据 i:%d %f\n",i,weights[i]);
      }
   }

   void initData(ActivationType activation){
      int nweights=channels*filters*ksize*ksize;
      float *weights=getWeights();
      int i;
      float scale = sqrt(2./(ksize*ksize*channels));
      if (activation == ActivationType.NORM_CHAN || activation == ActivationType.NORM_CHAN_SOFTMAX
      ||activation ==ActivationType.NORM_CHAN_SOFTMAX_MAXVAL) {
         for (i = 0; i < nweights; ++i)
            weights[i] = 1;   // rand_normal();
      }else{
         for (i = 0; i < nweights; ++i)
            weights[i] = scale*DnnUtils.randUniform/*!rand_uniform*/(-1, 1);   // rand_normal();
      }
   }

   float *getKernel(int index){
      float *weights=getWeights();
      return weights+index*ksize*ksize*channels;
   }

   float *getData(int filter,int channel){
      float *weights=getWeights();
      return weights+filter*ksize*ksize*self->channels+channel*ksize*ksize;
   }

   /**
   *
   */
   void rot180(float *input,float *output){
      int h,w;
      for(h=0;h<ksize;h++)
         for(w=0;w<ksize;w++)
            output[h*ksize+w]=input[(ksize-h-1)*ksize+ksize-w-1];
   }
   /**
   * 创建一个通道的所有filter数据，并旋转180
   * 把所有filters中的第一个通道的数据全部旋转180按行排列在一起。
   * 例如：filters=5 channels=256 size=3
   * 生成的数据是256x(5*3*3)的矩阵
   */
   float *createRot180Matrix(){
      int i,j;
      for(j=0;j<channels;j++){
         for(i=0;i<filters;i++){
            float *in=getData(i,j);
            float *out=rot180Data+j*filters*ksize*ksize+i*ksize*ksize;
            rot180(in,out);
         }
      }
      return rot180Data;
   }
};


