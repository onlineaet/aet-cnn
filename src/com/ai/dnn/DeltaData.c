#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <aet/time/Time.h>
#include "DeltaData.h"

impl$ DeltaData{

   DeltaData(int w,int h,int channels,int batch){
      super$(w,h,channels,batch);
      mean=calloc(channels,sizeof(float));
      variance=calloc(channels,sizeof(float));
   }

   DeltaData(int size,int batch){
      super$(size,batch);
      mean=NULL;
      variance=NULL;
   }

   //原型 mean_delta_cpu blas.h batchnorm_layer.c
   void  calcMean(float *variance){
      int filters=getChannels();
      int spatial=getWidth()*getHeight();
      float *delta = getDataArray();
      int i,j,k;
      for(i = 0; i < filters; ++i){
         mean[i] = 0;
         for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
               int index = j*filters*spatial + i*spatial + k;
               mean[i] += delta[index];
            }
         }
         mean[i] *= (-1./sqrt(variance[i] + .00001f));
      }
   }

   //原型 variance_delta_cpu blas.h batchnorm_layer.c
   void  calcVariance(NormData *normData){
      float *x = normData->getX();
      float *normMean=normData->getMean();
      float *normVariance = normData->getVariance();
      int filters = channels;
      int spatial = w*h;
      int i,j,k;
      for(i = 0; i < filters; ++i){
         variance[i] = 0;
         for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + i*spatial + k;
               variance[i] += dataArray[index]*(x[index] - normMean[i]);
            }
         }
         variance[i] *= -.5 * pow(normVariance[i] + .00001f, (float)(-3./2.));
      }
   }

   /**
   * 归一化误差数据
   */
   //原型 normalize_delta_cpu blas.h batchnorm_layer.c
   void normalize(NormData *normData){
      float *x = normData->getX();
      float *normMean = normData->getMean();
      float *normVariance= normData->getVariance();
      int filters=channels;
      int spatial=w*h;
      int f, j, k;
      for(j = 0; j < batch; ++j){
         for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + f*spatial + k;
               dataArray[index] = dataArray[index] * 1./(sqrt(normVariance[f] + .00001f))
                     + variance[f] * 2. * (x[index] - normMean[f]) / (spatial * batch) + mean[f]/(spatial*batch);
            }
         }
      }
   }

   public$ float *getMean(){
      return mean;
   }

   public$ float *getVariance(){
      return variance;
   }

   void checkMeanVariance(){
      float channels=getChannels();
      int i;
      if(5>3)
         return;
      for (i = 0; i < channels; ++i) {
         if(isnan(mean[i]) || isinf(mean[i])){
            printf("误差的平均值无效:%d %f\n",i,variance[i]);
            exit(0);
         }
         if(isnan(variance[i]) || isinf(variance[i])){
            printf("误差的方差无效:%d %f\n",i,variance[i]);
            exit(0);
         }
      }
   }

   DeltaData *clone(){
      int batch = getBatch();
      int size=     getSize();
      int w=    getWidth();
      int h=    getHeight();
      int c=    getChannels();
      DeltaData *newData=NULL;
      if(w>0 || h>0 || c>0)
         newData=new$ DeltaData(w,h,c,batch);
      else
         newData=new$ DeltaData(size,batch);
      if(!copy(newData)){
         newData->unref();
         a_error("克隆 DeletaData出错。");
      }
      if(mean){
         printf("deltedata.c clone mean batch:%d size:%d w:%d h:%d c:%d\n",batch,size,w,h,c);
         memcpy(newData->mean,mean,channels*sizeof(float));
      }
      if(variance){
         printf("deltedata.c clone variance batch:%d size:%d w:%d h:%d c:%d\n",batch,size,w,h,c);
         memcpy(newData->variance,variance,channels*sizeof(float));
      }
      return newData;
   }

   ~DeltaData(){
      if(mean)
         a_free(mean);
      if(variance)
         a_free(variance);
   }
};

