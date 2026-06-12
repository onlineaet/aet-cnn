#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/time/Time.h>
#include <aet/lang/System.h>
#include "ScaleData.h"
#include "DnnUtils.h"

static int calcCount=0;
static aint64 scaletime=0;

impl$ ScaleData{

   ScaleData(int size){
      self->size=size;
      scales=calloc(size,sizeof(float));
      scale_updates=calloc(size,sizeof(float));
   }

   void   init(float value){
      int i;
      for(i=0;i<size;i++)
         scales[i]=value;
   }

   float *getScales(){
      return scales;
   }

   float *getUpdates(){
      return scale_updates;
   }

   public$ ScaleData     *clone(){
      ScaleData *c=new$ ScaleData(size);
      memcpy(c->scales,scales,size*sizeof(float));
      memcpy(c->scale_updates,scale_updates,size*sizeof(float));
      if(ema){
         c->createEma();
         memcpy(c->ema,ema,size*sizeof(float));
      }
      return c;
   }
   /**
   *把scale_updates加到scales上
   */
   void   addUpdatesToScale(float alpha){
      int i;
      for(i=0;i<size;i++){
         if(isnan(alpha) || isnan(scale_updates[i]) || isnan(scales[i])){
            printf("ScaleData addUpdatesToScale %d %f scale_updates:%f scales:%f\n",i,alpha,scale_updates[i],scales[i]);
            abort();
         }
        // printf("scale is i:%d :%f %f\n",i,scales[i],scale_updates[i]);
         scales[i]+=alpha*scale_updates[i];
      }
   }

   /**
   * 缩放updates
   */
   void   scaleUpdates(float scale){
      int i;
      for(i=0;i<size;i++){
         scale_updates[i]*=scale;
         if(isnan(scale_updates[i])){
            printf("ScaleData scaleUpdates 出错了:%f\n",scale_updates[i]);
            abort();
         }
      }

   }

   //原型 backward_scale_cpu blas.h batchnorm_layer.c
   void backwardScale(NData *deltaData,NormData *normData){
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
      int spatial=w1*h1;
      int n = c1;

      float *x_norm=normData->getXNorm();
      float *delta=deltaData->getDataArray();

      int i,b,f;
      for(f = 0; f < n; ++f){
         float sum = 0;
         for(b = 0; b < batch; ++b){
            for(i = 0; i < spatial; ++i){
               int index = i + spatial*(f + n*b);
               sum += delta[index] * x_norm[index];
            }
         }
         scale_updates[f] += sum;
      }
   }


   void freeData(){
      if(scales){
         free(scales);
         scales=NULL;
      }
      if(scale_updates){
         free(scale_updates);
         scale_updates=NULL;
      }
   }

   //只有train才需要创建ema数据
   public$ void   createEma(){
      if(!ema)
         ema=calloc(size,sizeof(float));
   }

   public$ void   createUpdates(){
      if(!scale_updates)
         scale_updates=calloc(size,sizeof(float));
   }

   public$ int read(FILE *fp){
      return fread(scales,sizeof(float),size,fp);
   }

   public$ float *getEma(){
      return ema;
   }

   public$ int getSize(){
      return size;
   }

   ~ScaleData(){
      freeData();
   }
};

