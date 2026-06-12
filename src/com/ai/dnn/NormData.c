#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <bits/types.h>
#include <aet/lang/System.h>
#include <aet/time/Time.h>
#include <aet/mtcs/MtcsMem.h>
#include <aet/mtcs/MtcsSystem.h>
#include "NormData.h"
#include "DnnUtils.h"

#define ALIGN(x, mask)  (((x) + ((mask)-1)) & ~((mask)-1))

impl$ NormData{

   NormData(){
      batch=0;
      mean=NULL;
      variance=NULL;
      rolling_mean=NULL;
      rolling_variance=NULL;
      x=NULL;
      x_norm=NULL;
      w=0;
      h=0;
      channels=0;
      needThreads=System.getCpuThreads();//System.getCpuCores();
   }

   NormData(int w,int h,int channels,int batch){
      self();
      self->w=w;
      self->h=h;
      self->channels=channels;
      self->batch=batch;
      self->mean = NULL;//calloc(channels, sizeof(float));    //用于保存每个通道元素的平均值
      self->variance =NULL;// calloc(channels, sizeof(float));//用于保存每个通道元素的方差
      self->rolling_mean = calloc(channels, sizeof(float));
      self->rolling_variance = calloc(channels, sizeof(float));
      needThreads=System.getCpuCores();//System.getCpuThreads();//    System.getCpuCores();
      self->x_norm=NULL;//createData(w*h*channels);
      self->x=NULL;//createData(w*h*channels);
   }

   public$ void createX(){
      if(!x){
         x_norm=createData(w*h*channels);
         x=createData(w*h*channels);
      }
   }

   public$ void createMean(){
      if(!mean){
         mean=calloc(channels, sizeof(float));
      }
   }

   public$ void createVariance(){
      if(!variance){
         variance=calloc(channels, sizeof(float));
      }
   }


   float *createData(int elementSize){
      float *ret=DnnUtils.alignAlloc(batch*elementSize*sizeof(float));
      return ret;
   }

   /**
    * 复制ouputData的数据到x
    */
   void copyToX(NData *outputData){
      if(self->x==NULL){
         self->x=createData(w*h*channels);
      }
      int size=w*h*channels;
      memcpy(x,outputData->getDataArray(),batch*size*sizeof(float));
   }

   void copyToXNorm(NData*outputData){
      int batch=outputData->getBatch();
      int size=outputData->getSize();
      int width=outputData->getWidth();
      int height=outputData->getWidth();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      if(self->x_norm==NULL){
         self->x_norm=createData(w*h*channels);
      }
      memcpy(x_norm,outputData->getDataArray(),batch*size*sizeof(float));
   }

   /**
   * 计算outputData的平均值
   */
   //原型 mean_cpu blas.c
   void calcMean(float *x){
      int spatial=h*w;
      int filters = channels;
      float scale = 1./(batch * spatial);
      int i,j,k;
      for(i = 0; i < filters; ++i){
         mean[i] = 0;
         for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + i*spatial + k;
               mean[i] += x[index];
            }
         }
         mean[i] *= scale;
      }
   }

   /**
   * 计算方差
   */
   //原型 variance_cpu blas.c
   void calcVariance(float *x){
      int spatial=h*w;
      int filters = channels;
      float scale = 1./(batch * spatial - 1);
      int i,j,k;
      for(i = 0; i < filters; ++i){
         variance[i] = 0;
         for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
               int index = j*filters*spatial + i*spatial + k;
               variance[i] += pow((x[index] - mean[i]), 2);
            }
         }
         variance[i] *= scale;
      }
   }

   public$ void  calcVarianceAndMean(float *outputData){
      int h=getHeight();
      int w=getWidth();
      int channels=getChannels();
      int batch=getBatch();
      int spatial=h*w;
      int i,j,k;
      float temp[batch*channels];
      memset(temp,0,channels*batch*sizeof(float));
      float temp1[batch*channels];
      memset(temp1,0,channels*batch*sizeof(float));
      float temp2[batch*channels];
      memset(temp2,0,channels*batch*sizeof(float));
      //#pragma omp parallel for num_threads( needThreads ) private( i,j,k )
      for(i = 0; i < batch; ++i){
         float *item=outputData+i*spatial*channels;
         for(j = 0; j < channels; ++j){
            for(k = 0; k < spatial; k++){
               float value=item[j*spatial+k];
               temp[i*channels+j]+=value;
               temp1[i*channels+j]+=(value*value);
               temp2[i*channels+j]+=(-2*value);
            }
         }
      }
      float scale = 1./(batch * spatial);
      for(i = 0; i < channels; ++i){
         mean[i]=0;
         for(j = 0; j < batch; ++j){
            mean[i]+=temp[j*channels+i];
         }
         mean[i]*=scale;
      }
      scale = 1./(batch * spatial-1);
      float var_x[channels];
      for(i = 0; i < channels; ++i){
         float t1=0;
         float t2=0;
         for(j = 0; j < batch; ++j){
            t1+=temp1[j*channels+i];
            t2+=temp2[j*channels+i];
         }
         variance[i]=t1+t2*mean[i]+mean[i]*mean[i]*batch*spatial;
         variance[i]*=scale;
      }
   }


   void scal_cpus(int N, float ALPHA, float *X){
      int i;
      for(i = 0; i < N; ++i)
         X[i] *= ALPHA;
   }

   void axpy_cpus(int N, float ALPHA, float *X, float *Y){
      int i;
      for(i = 0; i < N; ++i)
         Y[i] += ALPHA*X[i];
   }
   /**
   * 计算普通和滚动（rolling）的均值和方差
   */
   void calcMeanAndVariance(NData *outputData,aboolean merge){
      float *outputDataArray=outputData->getDataArray();
      int batch=outputData->getBatch();
      int width=outputData->getWidth();
      int height=outputData->getHeight();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      if(!merge){
         calcMean(outputDataArray);
         calcVariance(outputDataArray);
      }else{
         calcVarianceAndMean(outputDataArray);
      }
   }

   /**
    * 缩放和累加滚动均值和方差
    * GPU 是 .99和.01
    */
   public$ void   scalAxpyRollMeanAndRollVariance(){
      scal_cpus(channels, .9, self->rolling_mean);
      axpy_cpus(channels, .1, mean, rolling_mean);
      scal_cpus(channels, .9, rolling_variance);
      axpy_cpus(channels, .1,variance, rolling_variance);
   }

   /**
   * 归一化输出数据
   * 归一化公式来自z-score标准化
   * A=v1,v2,v3,...,vn
   * 公式 v'=(vi-A的平均值)/σa(A的均方差)
   * 调用sqrt运行时间是70ms,如是提前做好sqrt的数组，变成20ms 提升3倍多速度。同时把归一化数据复制到x_norm
   * 提升速度。
   * avx的开平方根函数：_mm256_sqrt_ps
   * 1.在output改变前复制到x
   * 2.计算output的归一化数据
   * 3.outoput数据复制到xnorm
   * 4.outoupt乘加scales biases
   */
   void normalize(NData *outputData,float *scales,float *biases){
      float *outputDataArray=outputData->getDataArray();
      int batch=outputData->getBatch();
      int width=outputData->getWidth();
      int height=outputData->getHeight();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      int size = outputData->getSize();
      int b, f, i,j;
      int spatial=h*w;
      float sqrtVariance[channels];
      for(i=0;i<channels;i++)
         sqrtVariance[i]=1.0/(sqrt(variance[i])+ .000001f);
      //#pragma omp parallel for num_threads( needThreads ) private( b,f,i,j )
      for(b = 0; b < batch; ++b){
         float *outputData=outputDataArray+b*size;
         float *xnorm=x_norm+b*size;
         float *xc=x+b*size;
         float value=0;
         for(f = 0; f < channels; ++f){
            float *data=outputData+f*spatial;
            float *xnorm1=xnorm+f*spatial;
            float *xdata=xc+f*spatial;
            for(i = 0; i < spatial; i++){
               value =data[i];
               xdata[i]=value;
               value = (value - mean[f])*sqrtVariance[f];
               if(isnan(value)){
                  printf("normdata.c normalize 无效数据\n");
                  exit(0);
               }
               xnorm1[i]=value;
               data[i] = value*scales[f]+biases[f];
            }
         }
      }
   }

   /**
   * 用滚动平均值和滚动方差生成归一化数据
   * 1.在output改变前复制到x
   * 2.计算output的归一化数据
   * 3.outoupt乘加scales biases
   * normalize需要复制到x_norm,roll不需要
   */
   void normalizeByRoll(NData *outputData,float *scales,float *biases){
      float *outputDataArray=outputData->getDataArray();
      int batch=outputData->getBatch();
      int width=outputData->getWidth();
      int height=outputData->getHeight();
      int channels=outputData->getChannels();
      if(self->w!=width || self->h!=height || self->channels!=channels || self->batch!=batch){
         a_error("从输出数据生成规一化数据，但参数不正确。");
      }
      int size = outputData->getSize();
      int b, f, i,j;
      int spatial=h*w;
      float sqrtVariance[channels];
      for(i=0;i<channels;i++)
         sqrtVariance[i]=1.0/(sqrt(rolling_variance[i])+ .000001f);
      //#pragma omp parallel for num_threads( needThreads ) private( b,f,i,j )
      for(b = 0; b < batch; ++b){
         float *outputData=outputDataArray+b*size;
         float *xc=x+b*size;
         float value=0;
         for(f = 0; f < channels; ++f){
            float *data=outputData+f*spatial;
            float *xdata=xc+f*spatial;
            for(i = 0; i < spatial; i++){
               value =data[i];
               xdata[i]=value;
               value = (value - rolling_mean[f])*sqrtVariance[f];
               if(isnan(value)){
                  printf("normdata.c normalize 无效数据\n");
                  exit(0);
               }
               data[i] = value*scales[f]+biases[f];
            }
         }
      }
   }

   /**
   * 复制rolling_mean到mean
   */
   void copyToMean(float *src){
      if(mean==NULL)
         mean=malloc(channels*sizeof(float));
      memcpy(mean,src,channels*sizeof(float));
   }

   /**
   * 复制到方差
   */
   void copyToVarinace(float *src){
      if(variance==NULL)
         variance=malloc(channels*sizeof(float));
      memcpy(variance,src,channels*sizeof(float));
   }

   float *getMean(){
      return mean;
   }

   float *getVariance(){
      return variance;
   }

   float *getRollingMean(){
      return rolling_mean;
   }

   float *getRollingVariance(){
      return rolling_variance;
   }

   void resize(int w,int h){
      self->w=w;
      self->h=h;
      if(x){
         free(x);
         x=NULL;
      }
      if(x_norm){
         free(x_norm);
         x_norm=NULL;
      }
   }

   public$ int  getWidth(){
      return w;
   }

   public$ int  getHeight(){
      return h;
   }

   public$ int    getChannels(){
      return channels;
   }

   public$ int    getBatch(){
      return batch;
   }

   float *getX(int index,int ch){
      float *item=x+index*w*h*channels;
      return item+ch*w*h;
   }

   public$ float *getX(){
      return x;
   }

   public$ float *getXNorm(){
      return x_norm;
   }

   public$ float *getNorm(int index){
      return x_norm+index*w*h*channels;
   }


   ~NormData(){
      if(x){
         free(x);
         x=NULL;
      }
      if(x_norm){
         free(x_norm);
         x_norm=NULL;
      }
      if(mean){
         free(mean);
         mean=NULL;
      }
      if(variance){
         free(variance);
         variance=NULL;
      }

      if(rolling_mean){
         free(rolling_mean);
         rolling_mean=NULL;
      }

      if(rolling_variance){
         free(rolling_variance);
         rolling_variance=NULL;
      }
   }
};

